"""Redesigned H4: how does the camera's multiplexing level affect rare-feature survival?

Cameras: lenslet PSFs with k = 1 (a lens), 4, 16, 64, 256 spots, and the real DiffuserCam PSF.
Field of view: "full" (scene on the 2x padded grid, the real DiffuserCam setting) or "sensor".
Features: FIXED amplitudes in the scene (we compare cameras, so the feature must not depend on
the camera). Each feature's actual measurement d' under each camera is recorded.
Reconstruction: Tikhonov (linear), lambda chosen per camera by PSNR on the first 5 images.
DiffPIR with the ImageNet prior is the GPU follow-up (same script, --solver diffpir).

  python scripts/multiplexing_sweep.py --out results/multiplexing --sensor 64 --n-images 20
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rareprior.data import splits  # noqa: E402
from rareprior.data.loaders import center_square_resize, load_diffusercam_psf, load_diffusercam_test, to_tensor  # noqa: E402
from rareprior.features.insert import InsertConfig, ideal_tpr, sample_sites
from rareprior.features.library import template  # noqa: E402
from rareprior.metrics.image_quality import psnr  # noqa: E402
from rareprior.metrics.observer import add_detection, empirical_fpr, score_image, site_score, thresholds  # noqa: E402
from rareprior.operators.lensless import Lensless, lenslet_psf, prepare_psf  # noqa: E402
from rareprior.optics.psf import n_peaks, participation_ratio  # noqa: E402
from rareprior.solvers.cg import tikhonov  # noqa: E402
from rareprior.stats.bootstrap import paired_survival_bootstrap  # noqa: E402
from rareprior.utils.seeding import item_seed, rng, torch_gen  # noqa: E402


def cameras(S: int, layouts: int):
    cams = {"lenslet_k1": lenslet_psf((S, S), 1)}
    for k in (4, 16, 64, 256):
        for j in range(layouts):
            label = f"lenslet_k{k}" if j == 0 else f"lenslet_k{k}_layout{j}"
            cams[label] = lenslet_psf((S, S), k, seed=k if j == 0 else item_seed("layout", k, j))
    dc = prepare_psf(load_diffusercam_psf(), out_shape=(S, S)).mean(0, keepdim=True)
    cams["diffusercam"] = dc / dc.sum()
    return cams


def acquire(op, x, generator, photons=None):
    if photons is None:
        return op.add_noise(op.A(x), generator=generator), None
    # x is in [-1, 1]. Simulate counts from nonnegative radiance, then invert
    # the affine offset so the solver still receives a measurement of A(x).
    rate = photons * op.A((x + 1) / 2).clamp_min(0)
    y = 2 * torch.poisson(rate, generator=generator) / photons - op.A(torch.ones_like(x))
    return op.add_noise(y, generator=generator), float(rate.mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--sensor", type=int, default=64)
    ap.add_argument("--fov", default="full", choices=["full", "sensor"])
    ap.add_argument("--n-images", type=int, default=20)
    ap.add_argument("--layouts", type=int, default=1)
    ap.add_argument("--calibration-images", type=int, default=0,
                    help="Use this many separate dev scenes for lambda and observer thresholds.")
    ap.add_argument("--sigma", type=float, default=0.01)
    ap.add_argument("--photons", type=float, help="Incident count scale for optional shot noise.")
    ap.add_argument("--same-site", action="store_true",
                    help="Fit known-site thresholds from feature-off images on separate dev scenes.")
    ap.add_argument("--amplitudes", type=float, nargs="+", default=[0.2, 0.4])
    ap.add_argument("--lams", type=float, nargs="+", default=[1e-4, 1e-3, 1e-2, 1e-1])
    ap.add_argument("--iters", type=int, default=150)
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--solver", default="tikhonov", choices=["tikhonov", "diffpir"])
    ap.add_argument("--K", type=int, default=2, help="posterior samples (diffpir)")
    ap.add_argument("--nfe", type=int, default=100)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()
    if args.same_site and (not args.calibration_images or args.solver != "tikhonov"):
        ap.error("Same-site diagnostics require separate calibration and the linear solver.")
    if args.photons is not None and args.photons <= 0:
        ap.error("photons must be positive.")
    if args.layouts < 1 or args.calibration_images < 0:
        ap.error("Use positive layouts and nonnegative calibration-images.")
    if not 0 <= args.calibration_images < args.n_images <= len(splits.diffusercam_split()["dev"]):
        ap.error("Calibration and evaluation must fit inside the 30-scene dev pool.")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    S = args.sensor
    scene = 2 * S if args.fov == "full" else S
    ids = splits.diffusercam_split()["dev"][: args.n_images]
    rows_raw = load_diffusercam_test(ids)
    if args.solver == "diffpir":  # RGB scenes at the ImageNet prior's 256 px; scene must be 256
        assert scene == 256, "the ImageNet ADM prior is 256 px: use --sensor 128 with --fov full"
        imgs = {i: center_square_resize(to_tensor(rows_raw[i][1]), scene).float() for i in ids}
        from rareprior.solvers.registry import build_solver
        recon_fn, K = build_solver({"name": "diffpir", "prior": "imagenet", "K": args.K, "nfe": args.nfe},
                                   device=args.device)
    else:
        imgs = {i: center_square_resize(to_tensor(rows_raw[i][1]), scene).mean(0, keepdim=True).double()
                for i in ids}
    icfg = InsertConfig(n_features=6, n_controls=8, margin=max(8, scene // 16), min_sep=max(16, scene // 6),
                        ctrl_sep=max(8, scene // 12), amplitude_mode="fixed", amplitudes=tuple(args.amplitudes),
                        value_range=(-1.0, 1.0))
    cams = cameras(S, args.layouts)
    calib_ids = set(ids[:args.calibration_images])
    eval_ids = set(ids) - calib_ids
    all_rows, metrics, cam_info = [], [], {}
    for name, psf in cams.items():
        op = Lensless(psf, sigma=args.sigma, fov=args.fov).to(args.device if args.solver == "diffpir" else "cpu")
        t0 = time.time()
        # Lambda by PSNR on calibration scenes without features.
        best = (None, float("nan"))
        for lam in (args.lams if args.solver == "tikhonov" else []):
            ps = []
            for i in ids[:args.calibration_images or 5]:
                x = imgs[i][None]
                seed = ("lam", i) if calib_ids else ("lam", name, i)
                y, _ = acquire(op, x, torch_gen(*seed), args.photons)
                ps.append(float(psnr(tikhonov(op, y, lam, x.shape, iters=args.iters).float(), x.float())))
            if best[0] is None or np.mean(ps) > best[1]:
                best = (lam, float(np.mean(ps)))
        lam = best[0]
        for i in ids:
            x = imgs[i]
            # same features for every camera: the insertion RNG does not depend on the camera
            x_f, sites, dropped = sample_sites(x, op.cpu(), rng("multiplex", i), icfg)
            if args.solver == "diffpir":
                op.to(args.device)
            dev = args.device if args.solver == "diffpir" else "cpu"
            y, mean_counts = acquire(op, x_f[None].to(dev), torch_gen("multiplex-noise", i, device=dev),
                                     args.photons)
            if args.solver == "tikhonov":
                xh = tikhonov(op, y, lam, x_f[None].shape, iters=args.iters)
            else:
                seeds = [item_seed("multiplex-sample", name, i, k) for k in range(K)]
                xh = recon_fn(y, op, x_f[None].shape, seeds).detach().cpu()
            quality = float(psnr(xh.float(), x_f[None].float().expand_as(xh)).mean())
            metrics.append(dict(image_id=i, solver=name, psnr=quality, mean_expected_counts=mean_counts))
            r = []
            for k in range(xh.shape[0]):
                r += score_image(xh[k:k + 1], x[None], sites,
                                 dict(image_id=i, solver=name, severity=args.fov, sample=k))
            if args.same_site:
                off, _ = acquire(op, x[None], torch_gen("multiplex-noise", i), args.photons)
                x_off = tikhonov(op, off, lam, x[None].shape, iters=args.iters)[0]
                for site in sites:
                    if site.kind == "feature":
                        t = template(site.cls, site.size, site.variant)
                        score = site_score(x_off - x, t, site.cy, site.cx, site.sign)
                        r.append(dict(image_id=i, solver=name, severity=args.fov, sample=0,
                                      cy=site.cy, cx=site.cx, size=site.size, cls=site.cls,
                                      kind="cf_null", score=score))
            # location relative to the sensor footprint (the central S x S region of the scene)
            for row in r:
                c0, c1 = (scene - S) // 2, (scene + S) // 2
                row["in_footprint"] = bool(c0 <= row["cy"] < c1 and c0 <= row["cx"] < c1)
            all_rows += r
        cam_info[name] = dict(lam=lam, dev_psnr_for_lam=best[1], psnr=float(np.mean([m["psnr"] for m in metrics
                                                 if m["solver"] == name and m["image_id"] in eval_ids])),
                              participation_ratio=participation_ratio(psf[0]), n_peaks=n_peaks(psf[0]),
                              seconds=time.time() - t0)
        print(name, cam_info[name], flush=True)
    df = pd.DataFrame(all_rows)
    df.to_parquet(out / "rows.parquet")
    # thresholds per region: null scores inside and outside the sensor footprint differ a lot
    pd.DataFrame(metrics).to_csv(out / "image_metrics.csv", index=False)
    group = ["solver", "severity", "cls", "size", "in_footprint"]
    fit = df[df.image_id.isin(calib_ids)] if calib_ids else df
    eval_rows = df[df.image_id.isin(eval_ids)]
    if args.same_site:
        # Size and footprint strata limit sparsity in the ten-scene calibration set.
        group.remove("cls")
        fit = fit[fit.kind == "cf_null"].assign(kind="control")
        nulls = eval_rows[eval_rows.kind == "cf_null"].assign(kind="control")
    else:
        nulls = eval_rows
    taus = thresholds(fit, group=group)
    feats = add_detection(eval_rows, group=group, taus=taus)
    achieved = empirical_fpr(nulls, taus, group=group)
    achieved.to_csv(out / "fpr_by_cell.csv", index=False)
    if args.photons is None:
        feats["measurement_gaussian_tpr"] = feats.dprime_bke.map(ideal_tpr)
    small = feats[feats["size"].isin([3, 5])]
    summ = {}
    for name in cams:
        f = small[small.solver == name]
        summ[name] = dict(cam_info[name], n_small_scores=len(f), n_inside=int(f.in_footprint.sum()),
                          survival_small=float(f.detected.mean()),
                          measurement_gaussian_tpr=(float(f.measurement_gaussian_tpr.mean())
                                                    if args.photons is None else None),
                          fpr_eval_macro=float(achieved[achieved.solver == name].fp.mean()),
                          survival_small_in_footprint=float(f[f.in_footprint].detected.mean()),
                          survival_small_outside=float(f[~f.in_footprint].detected.mean())
                          if (~f.in_footprint).any() else None,
                          median_camera_dprime=float(f.dprime.median()),
                          ratio_median=float(f.ratio.median()))
    contrasts = {}
    for name in cams:
        if name != "diffusercam" and not calib_ids:
            contrasts[f"{name}_minus_diffusercam"] = paired_survival_bootstrap(
                df, name, "diffusercam", n_boot=args.n_boot, cell_filter=lambda d: d["size"].isin([3, 5]),
                tau_keys=["severity", "cls", "size", "in_footprint"])
    if calib_ids:
        # Conditional on thresholds and layouts. Resample paired scene clusters, not sites.
        from rareprior.stats.bootstrap import paired_metric_bootstrap
        image_means = small.groupby(["image_id", "solver"]).detected.mean().unstack().dropna()
        for name in cams:
            if name != "diffusercam":
                contrast = paired_metric_bootstrap(image_means[name].to_numpy(),
                                                   image_means["diffusercam"].to_numpy(), args.n_boot)
                contrasts[f"{name}_minus_diffusercam"] = dict(contrast, n_images=len(image_means))
    res = {"config": vars(args) | {"out": Path(args.out).name}, "cameras": summ,
           "calibration_ids": sorted(calib_ids), "evaluation_ids": sorted(eval_ids),
           "contrast_estimand": "mean of paired per-image rates, conditional on calibration" if calib_ids else
           "legacy macro weighting",
           "contrasts_small": contrasts}
    (out / "summary.json").write_text(json.dumps(res, indent=2, default=float))
    print(json.dumps(res, indent=2, default=float))


if __name__ == "__main__":
    main()
