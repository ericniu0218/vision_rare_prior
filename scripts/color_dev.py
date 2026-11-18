"""Matched measurement-energy color controls using Matisse's D65 cone transform.

Synthetic engineering demonstration. It measures template response in a
specified display/observer model. It estimates no human detection threshold.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import torch
import yaml

from rareprior.features.insert import embed
from rareprior.features.library import template
from rareprior.operators.blur import GaussianBlur
from rareprior.solvers.cg import tikhonov


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matisse", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    sys.path.insert(0, str(args.matisse.resolve()))
    from Simulated.Retina.RetinaModel import RetinaModel

    cfg = yaml.safe_load((args.matisse / "Experiment/Config/Default/LMS.yaml").read_text())
    cst = RetinaModel(cfg, "cpu").CST
    xyz = cst.linsRGB_to_CIEXYZ_matrix.double()
    luminance = xyz[1]
    rgb_to_lms = cst.linsRGB_to_LMS_matrix[:, :3].double()
    dirs = {"achromatic": torch.ones(3, dtype=torch.float64),
            "isoluminant_RG": torch.tensor([1.0, -float(luminance[0] / luminance[1]), 0.0],
                                          dtype=torch.float64)}
    op = GaussianBlur(31, 2.0, .02)
    generator = torch.Generator().manual_seed(72)
    rows = []
    for i in range(12):
        bg = 0.45 + .1 * torch.rand((1, 3, 64, 64), generator=generator, dtype=torch.float64)
        noise = op.add_noise(torch.zeros_like(bg), generator=generator)
        off = tikhonov(op, op.A(bg) + noise, .03, bg.shape)
        for size in (3, 5, 9):
            patch = embed(template("blob", size), 32, 32, (64, 64), 1, dtype=torch.float64)
            for label, direction in dirs.items():
                signal = patch * direction[None, :, None, None]
                signal *= (4 * op.sigma / op.A(signal).norm())
                if (bg + signal).min() < 0 or (bg + signal).max() > 1:
                    raise ValueError("Stimulus outside display gamut.")
                on = tikhonov(op, op.A(bg + signal) + noise, .03, bg.shape)
                projected = ((on - off) * patch).sum(dim=(0, 2, 3)) / patch.square().sum()
                expected = (signal * patch).sum(dim=(0, 2, 3)) / patch.square().sum()
                rows.append(dict(scene=i, size=size, stimulus=label,
                                 measurement_dprime=float(op.A(signal).norm() / op.sigma),
                                 input_Y=float(expected @ luminance), recovered_Y=float(projected @ luminance),
                                 input_LMS=(expected @ rgb_to_lms).tolist(),
                                 recovered_LMS=(projected @ rgb_to_lms).tolist(),
                                 linear_rgb_response=float(projected @ expected / expected.square().sum())))
    summary = dict(matisse_commit=subprocess.check_output(
        ["git", "-C", str(args.matisse), "rev-parse", "HEAD"], text=True).strip(),
        kind="synthetic linear-RGB paired reconstruction and display model", n_scenes=12, rows=rows)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
