"""MRI pipeline demo on a phantom (mridata.org is unreachable from the build container).

Multi-coil SENSE with SigPy birdcage coil maps, variable-density Poisson-disc masks
(sigpy.mri.poisson), small lesion-like features inserted with the background phase,
then reconstruction with: zero-filled adjoint, Tikhonov (CG), and SigPy L1-wavelet.
Survival vs acceleration R. Fixed feature amplitude (the acquisition varies), and
each feature's measurement d' is recorded.

This checks the pipeline only. Shepp-Logan is piecewise constant, which flatters
L1-wavelet; real conclusions need raw knee k-space (plan Section 8.3).

  python scripts/mri_phantom_demo.py --out results/mri_phantom --n-images 12
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rareprior.features.insert import InsertConfig, sample_sites  # noqa: E402
from rareprior.metrics.observer import add_detection, score_image  # noqa: E402
from rareprior.operators.mri import Sense, poisson_mask, to_complex, to_real2  # noqa: E402
from rareprior.solvers.cg import tikhonov  # noqa: E402
from rareprior.utils.seeding import rng, torch_gen  # noqa: E402


def phantom(n: int, seed: int):
    import sigpy as sp
    img = sp.shepp_logan((n, n)).real.astype(np.float64)
    r = np.random.default_rng(seed)
    yy, xx = np.meshgrid(np.linspace(-1, 1, n), np.linspace(-1, 1, n), indexing="ij")
    phase = 0.6 * r.standard_normal() * xx + 0.6 * r.standard_normal() * yy  # smooth background phase
    img = img * (0.85 + 0.3 * r.random())  # mild intensity variation between "subjects"
    z = img * np.exp(1j * phase)
    return torch.from_numpy(z), torch.from_numpy(phase)


def l1wavelet(y, maps, lam):
    import sigpy.mri as mr
    return torch.from_numpy(np.asarray(mr.app.L1WaveletRecon(y.numpy(), maps.numpy(), lamda=lam,
                                                             max_iter=60, show_pbar=False).run()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=128)
    ap.add_argument("--coils", type=int, default=8)
    ap.add_argument("--n-images", type=int, default=12)
    ap.add_argument("--accels", type=float, nargs="+", default=[2, 4, 6, 8])
    ap.add_argument("--sigma", type=float, default=0.01)
    ap.add_argument("--amplitudes", type=float, nargs="+", default=[0.1, 0.2])
    args = ap.parse_args()
    import sigpy.mri as mr
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    n = args.n
    maps = torch.from_numpy(mr.birdcage_maps((args.coils, n, n)).astype(np.complex128))
    icfg = InsertConfig(n_features=6, n_controls=8, margin=24, min_sep=24, ctrl_sep=12,
                        amplitude_mode="fixed", amplitudes=tuple(args.amplitudes), value_range=None,
                        sizes=(3, 5, 9))
    rows, info = [], {}
    for R in args.accels:
        mask = torch.from_numpy(poisson_mask((n, n), R, calib=16, seed=int(R)))
        op = Sense(maps, mask, sigma=args.sigma)
        lam_l1 = 2e-3
        ps = {"zero_filled": [], "tikhonov": [], "l1wavelet": []}
        for i in range(args.n_images):
            z, phase = phantom(n, i)
            x = to_real2(z[None])[0]
            # features restricted to the object (magnitude > 0.05) is natural for lesions; here anywhere inside
            x_f, sites, _ = sample_sites(x, op, rng("mri", i), icfg, phase=phase)
            y = op.add_noise(op.A(x_f[None]), generator=torch_gen("mri-noise", i, R))
            recs = {
                "zero_filled": op.AT(y, x_shape=x_f[None].shape),
                "tikhonov": tikhonov(op, y, 1e-2, x_f[None].shape, iters=60),
                "l1wavelet": to_real2(l1wavelet(y[0], maps, lam_l1)[None]),
            }
            for name, xh in recs.items():
                err = (to_complex(xh) - to_complex(x_f[None])).abs().pow(2).mean()
                ps[name].append(float(10 * torch.log10(to_complex(x_f[None]).abs().max() ** 2 / err)))
                rows += score_image(xh, x[None], sites, dict(image_id=i, solver=name, severity=f"R{R:g}", sample=0),
                                    phase=phase)
        info[f"R{R:g}"] = {"accel_actual": float(mask.numel() / mask.sum()),
                           **{f"psnr_{k}": float(np.mean(v)) for k, v in ps.items()}}
        print(R, info[f"R{R:g}"], flush=True)
    df = pd.DataFrame(rows)
    df.to_parquet(out / "rows.parquet")
    feats = add_detection(df)
    small = feats[feats["size"].isin([3, 5])]
    surv = small.groupby(["severity", "solver"]).detected.mean().unstack()
    dmed = small.groupby("severity").dprime.median()
    res = {"config": vars(args), "acquisition": info, "survival_small": surv.round(3).to_dict(),
           "median_dprime_small": dmed.round(2).to_dict()}
    (out / "summary.json").write_text(json.dumps(res, indent=2))
    print(surv.round(3))
    print(json.dumps(res["acquisition"], indent=2))


if __name__ == "__main__":
    main()
