"""Make figures from the audited optics and MRI development summaries."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def optics(root, out):
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.65), sharey=True, layout="constrained")
    for ax, run, title in zip(axes, ("optics_read", "optics_shot"),
                              ("A. Gaussian read noise", "B. Shot + read noise"), strict=True):
        data = json.loads((root / run / "summary.json").read_text())
        for j, (cam, r) in enumerate(data["cameras"].items()):
            marker = "s" if cam == "diffusercam" else "*" if cam == "lenslet_k1" else "o"
            for col, label, color in (("survival_small_in_footprint", "Inside footprint", "#156779"),
                                       ("survival_small_outside", "Outside footprint", "#ab4b24")):
                ax.scatter(r["psnr"], r[col], c=color, s=65 if marker != "o" else 23,
                           marker=marker, alpha=.84, label=label if j == 0 else None)
            if cam in ("lenslet_k1", "diffusercam"):
                ax.annotate("Lens" if cam == "lenslet_k1" else "DiffuserCam PSF",
                            (r["psnr"], r["survival_small_in_footprint"]),
                            xytext=(5, 7) if cam == "lenslet_k1" else (-3, 23),
                            textcoords="offset points", fontsize=8, ha="left" if cam == "lenslet_k1" else "right",
                            arrowprops=None if cam == "lenslet_k1" else dict(arrowstyle="-", lw=.5))
        ax.set(title=title, xlabel="Whole-scene PSNR (dB)", ylim=(0, .74))
        ax.grid(alpha=.18)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("Detected fraction of small sites")
    axes[0].legend(loc="upper right", frameon=False, fontsize=8, markerscale=.8)
    fig.suptitle("Lensless simulation | 20 evaluation scenes, 10 separate calibration scenes", fontsize=12)
    fig.savefig(out / "optics_development.png", dpi=180)
    plt.close(fig)


def mri(root, out):
    data = json.loads((root / "mri_raw_dev.json").read_text())
    fig, axes = plt.subplots(1, 2, figsize=(8.5, 3.35), layout="constrained")
    for solver, name, color in (("adjoint", "Adjoint", "#ab4b24"), ("sense", "SENSE", "#156779"),
                                ("l1wavelet", "L1 wavelet", "#644e84")):
        rows = data["rows"]
        accel = [r["actual_accel"] for r in rows]
        quality = [r["solver"][solver]["mag_nrmse_vs_full_sense"] for r in rows]
        amplitude = [r["solver"][solver]["paired_amplitude_response"] for r in rows]
        axes[0].scatter(accel, quality, color=color, label=name, s=23, alpha=.8)
        axes[1].scatter([a for a in accel for _ in (0, 1)], sum(amplitude, []), color=color, s=23, alpha=.65)
    axes[0].legend(frameon=False, fontsize=8)
    axes[0].set_ylabel("Magnitude NRMSE vs full SENSE")
    axes[1].set_ylabel("Paired amplitude response")
    axes[1].axhline(1, color="#595959", lw=.8, ls="--")
    for ax in axes:
        ax.set(xlabel="Measured acceleration", xticks=[4, 8], xlim=(3.1, 8.9))
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(alpha=.18)
    fig.suptitle("Stanford knee development case | Three slices from one acquisition", fontsize=12)
    fig.savefig(out / "raw_knee_development.png", dpi=180)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    optics(args.results, args.out)
    mri(args.results, args.out)


if __name__ == "__main__":
    main()
