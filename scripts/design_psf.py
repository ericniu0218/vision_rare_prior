"""Optimise lensless PSFs for three objectives, starting from the real DiffuserCam PSF.

Outputs (in --out): psf_<name>.npy for diffusercam, mse, mi, detect (free-form) and
optionally phase-mask variants; summary.json with every objective, multiplexing
metrics for every PSF; psfs.png.

  python scripts/design_psf.py --out artifacts/psf --size 128 --sigma 0.01
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rareprior.data import splits  # noqa: E402
from rareprior.optics.psf import (FreeFormPSF, PhaseMaskPSF, as_numpy_image, optimize_psf,  # noqa: E402
                                  scene_power_spectrum, summarize, template_spectra)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--size", type=int, default=256)
    ap.add_argument("--sigma", type=float, default=0.01)
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--lr", type=float, default=0.05)
    ap.add_argument("--n-scenes", type=int, default=200)
    ap.add_argument("--phase-mask", action="store_true")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(0)

    from rareprior.data.loaders import center_square_resize, load_diffusercam_psf, load_diffusercam_test, to_tensor
    from rareprior.operators.lensless import prepare_psf

    S = args.size
    psf0 = prepare_psf(load_diffusercam_psf(), out_shape=(S, S)).mean(0)
    psf0 = psf0 / psf0.sum()
    # power spectrum from DiffuserCam test-split scenes that are NOT in our dev/test subsets
    sp = splits.diffusercam_split()
    pool = [i for i in range(999) if i not in set(sp["dev"]) | set(sp["test"])][: args.n_scenes]
    rows = load_diffusercam_test(pool)
    scenes = torch.stack([center_square_resize(to_tensor(rows[i][1]), S).mean(0) for i in pool])
    pad = (2 * S, 2 * S)
    P = scene_power_spectrum(scenes, pad)
    T2 = template_spectra(pad)
    sigma2 = args.sigma ** 2

    psfs = {"diffusercam": psf0.double()}
    hist = {}
    for name in ("mse", "mi", "detect"):
        param = FreeFormPSF((S, S), init=psf0.double())
        h, hh = optimize_psf(param, name, pad, P, sigma2, T2, steps=args.steps, lr=args.lr)
        psfs[name] = h
        hist[name] = hh
        print(name, hh[0], hh[-1], flush=True)
    if args.phase_mask:
        for name in ("mi", "detect"):
            param = PhaseMaskPSF((S, S))
            h, hh = optimize_psf(param, name, pad, P, sigma2, T2, steps=args.steps // 2, lr=args.lr)
            psfs[f"{name}_phase"] = h
            hist[f"{name}_phase"] = hh
            print(name + "_phase", hh[0], hh[-1], flush=True)

    summary = {k: summarize(h, pad, P, sigma2, T2) for k, h in psfs.items()}
    for k, h in psfs.items():
        np.save(out / f"psf_{k}.npy", h.numpy())
    (out / "summary.json").write_text(json.dumps({"config": vars(args), "psfs": summary, "history": hist},
                                                 indent=2))
    print(json.dumps(summary, indent=2))
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, len(psfs), figsize=(3 * len(psfs), 3.2))
        for ax, (k, h) in zip(axes, psfs.items()):
            ax.imshow(as_numpy_image(h) ** 0.5, cmap="magma")
            ax.set_title(f"{k}\nPR={summary[k]['participation_ratio']:.3f}", fontsize=9)
            ax.axis("off")
        fig.tight_layout()
        fig.savefig(out / "psfs.png", dpi=150)
    except Exception as e:  # noqa: BLE001
        print("plot skipped:", e)


if __name__ == "__main__":
    main()
