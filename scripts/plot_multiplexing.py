"""Figure: small-feature survival vs PSNR across cameras (multiplexing sweep)."""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

runs = sys.argv[1:-1]
out = Path(sys.argv[-1])
fig, axes = plt.subplots(1, len(runs), figsize=(5.2 * len(runs), 4), squeeze=False)
order = ["lenslet_k1", "lenslet_k4", "lenslet_k16", "lenslet_k64", "lenslet_k256", "diffusercam"]
labels = ["lens", "k=4", "k=16", "k=64", "k=256", "DiffuserCam"]
for ax, run in zip(axes[0], runs):
    r = json.loads((Path(run) / "summary.json").read_text())
    cams = r["cameras"]
    ps = [cams[c]["psnr"] for c in order]
    sv = [cams[c]["survival_small"] for c in order]
    ax2 = ax.twinx()
    ax.plot(range(6), sv, "o-", color="#c0392b", label="small-feature survival")
    ax2.plot(range(6), ps, "s--", color="#2c3e50", label="PSNR")
    ax.set_xticks(range(6), labels, rotation=30)
    ax.set_ylabel("survival (TPR @ 5% FPR)", color="#c0392b")
    ax2.set_ylabel("PSNR (dB)", color="#2c3e50")
    ax.set_ylim(0, 1.05)
    ax.set_title(f"field of view: {r['config']['fov']} (Tikhonov, {r['config']['n_images']} dev scenes)", fontsize=10)
    ax.set_xlabel("more multiplexing →")
fig.tight_layout()
fig.savefig(out, dpi=160)
print("saved", out)
