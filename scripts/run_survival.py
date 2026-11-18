"""Insert features -> simulate measurement -> reconstruct -> score every site.

One output file per image (resumable, shardable across Kaggle sessions):
  <out>/rows_<id>.parquet     site scores (+ detector scores when K > 1)
  <out>/metrics_<id>.json     PSNR / SSIM (/ LPIPS) per posterior sample
  <out>/crops_<id>.npz        64x64 crops around every site, fp16
  <out>/manifest.json

Usage:
  python scripts/run_survival.py configs/experiment/<name>.yaml [key=value ...]
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
from rareprior.detect.scores import residual_score, semantic_score, variance_score, window  # noqa: E402
from rareprior.diffusion.loader import download_checkpoint, sha256_file
from rareprior.features.insert import InsertConfig, embed, sample_sites  # noqa: E402
from rareprior.features.library import template  # noqa: E402
from rareprior.metrics.image_quality import psnr, ssim  # noqa: E402
from rareprior.metrics.observer import score_image, site_score  # noqa: E402
from rareprior.solvers.registry import build_solver  # noqa: E402
from rareprior.utils.config import load_config  # noqa: E402
from rareprior.utils.manifest import atomic_output, git_sha, stage_dir, write_manifest  # noqa: E402
from rareprior.utils.seeding import item_seed, rng, torch_gen  # noqa: E402


# ----------------------------------------------------------------------------- data
def synthetic_image(idx: int, size: int = 256, channels: int = 3) -> torch.Tensor:
    """Smooth random 'natural-ish' image with a 1/f^2 spectrum (CPU smoke tests only)."""
    g = torch.Generator().manual_seed(item_seed("synthetic", idx))
    f = torch.fft.fftfreq(size)
    fr = torch.sqrt(f[:, None] ** 2 + f[None, :] ** 2).clamp_min(1.0 / size)
    out = []
    for _ in range(channels):
        ph = torch.randn(size, size, generator=g) + 1j * torch.randn(size, size, generator=g)
        img = torch.fft.ifft2(ph / fr).real
        img = (img - img.mean()) / (img.std() * 4)
        out.append(img)
    x = torch.stack(out)
    return x.clamp(-0.8, 0.8)


def image_ids(cfg) -> list[int]:
    d = cfg["data"]
    kind = d["kind"]
    split = cfg["split"]
    splits.require_development(split)
    if "image_ids" in cfg:
        ids = list(cfg["image_ids"])
    elif kind == "ffhq_hf":
        ids = splits.FFHQ["dev"][:cfg.get("n_images", 30)]
    elif kind == "diffusercam_lensed":
        ids = splits.diffusercam_split()["dev"][:cfg.get("n_images", 30)]
    else:
        ids = list(range(cfg.get("n_images", 10)))
    splits.require_development(split, ids, kind)
    if not ids:
        raise ValueError("No development images selected.")
    return ids


def get_images(cfg, ids):
    d = cfg["data"]
    kind = d["kind"]
    size = d.get("size", 256)
    if kind == "synthetic":
        return lambda i: synthetic_image(i, size, d.get("channels", 3))
    if kind == "ffhq_hf":
        from rareprior.data.loaders import load_ffhq
        cache = load_ffhq(ids, repo=d.get("repo", "merkol/ffhq-256"), size=size, revision=d.get("revision"))
        return lambda i: cache[i]
    if kind == "diffusercam_lensed":
        from rareprior.data.loaders import center_square_resize, load_diffusercam_test, to_tensor
        rows = load_diffusercam_test(ids)
        return lambda i: center_square_resize(to_tensor(rows[i][1]), size)
    if kind == "folder":
        from PIL import Image

        from rareprior.data.loaders import center_square_resize, to_tensor
        root = Path(d["root"])
        files = sorted(root.glob("*.png"))
        return lambda i: center_square_resize(to_tensor(Image.open(files[i]).convert("RGB")), size)
    raise ValueError(kind)


def build_operator(cfg, device):
    o = cfg["operator"]
    if o["kind"] == "blur":
        from rareprior.operators.blur import GaussianBlur
        op = GaussianBlur(o.get("kernel_size", 61), o["std"], o.get("sigma", 0.05),
                          boundary=o.get("boundary", "reflect")).to(device)
        backend = o.get("blur_backend", "fft")
        if backend not in {"fft", "spatial"}:
            raise ValueError(f"Unknown blur backend: {backend}")
        op.use_fft = backend == "fft"
        return op
    if o["kind"] == "lensless":
        from rareprior.operators.lensless import Lensless, prepare_psf
        if o.get("psf_npy"):
            psf = torch.from_numpy(np.load(o["psf_npy"]))
            if psf.ndim == 2:
                psf = psf[None]
        else:
            from rareprior.data.loaders import load_diffusercam_psf
            psf = prepare_psf(load_diffusercam_psf(), out_shape=(o.get("size", 256),) * 2)
            if o.get("grayscale_psf", True):
                psf = psf.mean(0, keepdim=True)
        return Lensless(psf, sigma=o.get("sigma", 0.01)).to(device)
    raise ValueError(o["kind"])


# ----------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("config")
    ap.add_argument("overrides", nargs="*")
    args = ap.parse_args()
    cfg = load_config(args.config, args.overrides)
    device = cfg.get("device", "cuda" if torch.cuda.is_available() else "cpu")
    torch.set_grad_enabled(True)

    ids = image_ids(cfg)
    shard, nsh = int(cfg.get("shard", 0)), int(cfg.get("num_shards", 1))
    if not 0 <= shard < nsh:
        raise ValueError("shard must be in [0, num_shards).")
    local_ids = ids[shard::nsh]
    full_ids = set(ids[:int(cfg.get("save_full", 0))])

    run_cfg = {k: v for k, v in cfg.items() if k not in ("shard", "num_shards", "device")}
    solver_cfg = dict(cfg["solver"])
    if solver_cfg["name"] in {"dps", "diffpir"} and not solver_cfg.get("checkpoint"):
        solver_cfg["checkpoint"] = download_checkpoint(solver_cfg.get("prior", "ffhq"))
    files = [solver_cfg.get(k) for k in ("weights", "checkpoint")]
    files.extend([cfg.get("insert", {}).get("power_spectrum"), cfg["operator"].get("psf_npy")])
    run_cfg["input_sha256"] = {
        role: sha256_file(path)
        for role, path in zip(("weights", "checkpoint", "power_spectrum", "psf"), files, strict=True)
        if path
    }
    run_cfg["image_ids"] = ids
    run_cfg["code_commit"] = git_sha()
    run_cfg["torch_version"] = str(torch.__version__)
    run_cfg["backend"] = torch.device(device).type
    out = stage_dir(cfg.get("out_root", "artifacts"), f"survival/{cfg['name']}", run_cfg)
    write_manifest(out, run_cfg, split=cfg["split"])
    print(f"output: {out}")

    def complete(i):
        files = [f"rows_{i}.parquet", f"metrics_{i}.json", f"crops_{i}.npz"]
        if i in full_ids:
            files.append(f"full_{i}.npz")
        return all((out / name).is_file() and (out / name).stat().st_size > 0 for name in files)

    ids = [i for i in local_ids if not complete(i)]
    if not ids:
        return
    get = get_images(cfg, ids)
    op = build_operator(cfg, device)
    recon, K = build_solver(solver_cfg, device)
    sname = solver_cfg.get("label", solver_cfg["name"])
    icfg = InsertConfig(**{k: tuple(v) if isinstance(v, list) else v for k, v in cfg.get("insert", {}).items()})
    sev = cfg.get("severity", "S1")
    want_cf = bool(cfg.get("counterfactual", False))

    for n, img_id in enumerate(ids):
        f_rows = out / f"rows_{img_id}.parquet"
        f_rows.unlink(missing_ok=True)  # an incomplete legacy image must be republished last
        t0 = time.time()
        x = get(img_id).to(device)
        x_f, sites, dropped = sample_sites(x.cpu(), op.cpu(), rng(cfg["arm"], img_id, sev, "insert"), icfg)
        op.to(device)
        x_f = x_f.to(device)
        g = torch_gen(cfg["arm"], img_id, sev, "noise", device=device)
        y_clean = op.A(x_f[None])
        noise = op.add_noise(torch.zeros_like(y_clean), generator=g)
        y = y_clean + noise
        seeds = [item_seed(cfg["arm"], img_id, sev, sname, "sample", k) for k in range(K)]
        xh = recon(y, op, x_f[None].shape, seeds).detach()
        meta = dict(image_id=img_id, solver=sname, severity=sev, dropped=dropped)
        rows = []
        for k in range(xh.shape[0]):
            rows += score_image(xh[k:k + 1].cpu(), x[None].cpu(), sites, dict(meta, sample=k))
        df = pd.DataFrame(rows)

        # detector scores (per site; residual per sample then averaged)
        feat_sites = [s for s in sites if s.kind == "feature"] + [s for s in sites if s.kind == "control"]
        null_maps = []
        gn = torch_gen("null", img_id, device=device)
        for _ in range(8):
            null_maps.append(op.AT(op.add_noise(torch.zeros_like(y_clean), generator=gn), x_shape=x_f[None].shape))
        det = []
        for s in feat_sites:
            r = 16
            null_e = float(np.mean([float((window(m, s.cy, s.cx, r) ** 2).sum()) for m in null_maps]))
            res = np.mean([residual_score(op, y, xh[k:k + 1], s.cy, s.cx, r, null_e) for k in range(xh.shape[0])])
            d = dict(cy=s.cy, cx=s.cx, det_residual=res)
            if xh.shape[0] > 1:
                d["det_variance"] = variance_score(xh.cpu(), s.cy, s.cx, r)
                use_lpips = bool(cfg.get("lpips", False))
                key = "det_semantic" if use_lpips else "det_pixel_pair"
                d[key] = semantic_score(xh.cpu(), s.cy, s.cx, crop=32, use_lpips=use_lpips)
            det.append(d)
        df = df.merge(pd.DataFrame(det), on=["cy", "cx"], how="left")

        if want_cf:  # counterfactual null: same noise, no features
            y_off = op.A(x[None]) + noise
            xh0 = recon(y_off, op, x[None].shape, seeds).detach().cpu()
            cf = []
            xh_cpu = xh.cpu()
            for s in sites:
                if s.kind != "feature":
                    continue
                t = template(s.cls, s.size, s.variant)
                target = embed(t - t.mean(), s.cy, s.cx, x.shape[-2:], x.shape[0], dtype=x.dtype).to(device)
                measured_target = op.A(target) * s.sign
                measured_norm = measured_target.abs().norm().clamp_min(1e-12)
                measurement_on = float((y * measured_target.conj()).real.sum() / measured_norm)
                measurement_off = float((y_off * measured_target.conj()).real.sum() / measured_norm)
                for k in range(xh0.shape[0]):
                    match = ((df.kind == "feature") & (df["sample"] == k) & (df.cy == s.cy) & (df.cx == s.cx))
                    df.loc[match, "no_clean_score"] = site_score(xh_cpu[k], t, s.cy, s.cx, s.sign)
                    df.loc[match, "measurement_score"] = measurement_on
                    cf.append(dict(meta, sample=k, kind="cf_null", cls=s.cls, size=s.size, variant=s.variant,
                                   sign=s.sign, cy=s.cy, cx=s.cx,
                                   score=site_score(xh0[k] - x.cpu(), t, s.cy, s.cx, s.sign),
                                   no_clean_score=site_score(xh0[k], t, s.cy, s.cx, s.sign),
                                   measurement_score=measurement_off))
            df = pd.concat([df, pd.DataFrame(cf)], ignore_index=True)

        m = {"image_id": img_id, "solver": sname, "severity": sev, "seconds": time.time() - t0,
             "psnr": psnr(xh.cpu().float(), x_f[None].cpu().float().expand_as(xh)).tolist(),
             "ssim": ssim(xh.cpu().double(), x_f[None].cpu().double().expand_as(xh)).tolist(),
             "dropped": dropped}
        if cfg.get("lpips", False):
            from rareprior.metrics.image_quality import lpips_distance
            m["lpips"] = lpips_distance(xh.cpu(), x_f[None].cpu().expand_as(xh)).tolist()
        with atomic_output(out / f"metrics_{img_id}.json") as temp:
            temp.write_text(json.dumps(m, allow_nan=False))
        crops = {}
        for j, s in enumerate(sites):
            c = window(torch.nn.functional.pad(xh.cpu(), (32, 32, 32, 32)), s.cy + 32, s.cx + 32, 32)
            crops[f"site{j}"] = c.numpy().astype(np.float16)
        with atomic_output(out / f"crops_{img_id}.npz") as temp:
            np.savez_compressed(temp, **crops)
        if img_id in full_ids:
            with atomic_output(out / f"full_{img_id}.npz") as temp:
                np.savez_compressed(temp, x=x.cpu().numpy(), x_f=x_f.cpu().numpy(),
                                    y=y.cpu().numpy(), xh=xh.cpu().numpy().astype(np.float16))
        if df.empty or not np.isfinite(df.score).all():
            raise ValueError(f"Invalid scores for development image {img_id}")
        with atomic_output(f_rows) as temp:
            df.to_parquet(temp)
        print(f"[{n + 1}/{len(ids)}] image {img_id}: {time.time() - t0:.1f}s, "
              f"PSNR {np.mean(m['psnr']):.2f}, dropped {dropped}", flush=True)


if __name__ == "__main__":
    main()
