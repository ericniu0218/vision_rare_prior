"""Recompute the best lenslet count from Kabuli et al. supplementary MI arrays.

Uses the published estimates from the read-noise bead study. It does not train
the entropy estimator or reproduce the physical imaging experiment.
"""
from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

import numpy as np

PIN = "f26873c2b58b82d84353bab2bf6edd478c5892a8"
NAMES = ["one", "two", "three", "four", "five", "six", "seven", "eight", "nine"]
FRACTIONS = [.008, .009, .01, .012, .013, .014, .015, .02, .025, .03, .04,
             .05, .06, .07, .08, .09, .10, .11, .12, .13, .14, .15]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    commit = subprocess.check_output(["git", "-C", str(args.repo), "rev-parse", "HEAD"], text=True).strip()
    if commit != PIN:
        raise ValueError(f"Use LenslessInfoDesign commit {PIN}.")
    data = args.repo / "tradeoff_analysis/mi_estimates/bead_mi_estimates"
    rows = []
    for fraction in FRACTIONS:
        values = []
        for label in NAMES:
            name = f"pixelcnn_mi_estimate_{fraction}_sparsity_100_photons_5_sigma_{label}_psf"
            arr = np.load(data / f"{name}_0.001_lr_20_patience_500_steps_per_epoch.npy", allow_pickle=False)
            j = int(np.argmin(arr[0]))  # choice in the published S4 notebook
            values.append([float(col[j]) for col in arr])
        rows.append(dict(bead_fraction=fraction, best_lenslets=1 + int(np.argmax(np.asarray(values)[:, 0])),
                         mi_by_lenslet=values))
    summary = dict(source_commit=commit, units="bits per pixel", mean_photon_count=100, read_noise_sd=5, rows=rows)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
