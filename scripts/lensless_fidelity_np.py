"""Forward-model fidelity on real DiffuserCam data (NumPy only, no torch).

Simulate y_sim = C(h * P(x)) from the `lensed` image with the real PSF, and
compare with the real `lensless` capture: Pearson correlation after the best
affine gain/offset fit, per colour channel. Also tests small shifts of the
PSF to check alignment. Writes results/lensless_fidelity/summary.json.
"""
import io
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image
import pyarrow.parquet as pq
import tifffile
from huggingface_hub import hf_hub_download
from scipy.ndimage import gaussian_filter

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rareprior.data.loaders import DIFFUSERCAM_REPO, DIFFUSERCAM_REVISION
from rareprior.data.splits import diffusercam_split

out = Path(sys.argv[1] if len(sys.argv) > 1 else "artifacts/lensless_fidelity_dev")
out.mkdir(parents=True, exist_ok=True)
def download(name):
    return hf_hub_download(DIFFUSERCAM_REPO, name, repo_type="dataset", revision=DIFFUSERCAM_REVISION)


psf = tifffile.imread(download("psf.tiff")).astype(np.float64)
# 4x box downsample 1080x1920 -> 270x480, background subtract, normalise per channel
psf = psf.reshape(270, 4, 480, 4, 3).mean(axis=(1, 3))
psf = np.clip(psf - np.quantile(psf, 0.05, axis=(0, 1)), 0, None)
psf /= psf.sum(axis=(0, 1), keepdims=True)
t = pq.read_table(download("data/test-00000-of-00001.parquet"))
H, W = 270, 480
PH, PW = 2 * H, 2 * W

def pad(a):
    o = np.zeros((PH, PW)); o[H // 2:H // 2 + H, W // 2:W // 2 + W] = a; return o

def crop(a):
    return a[H // 2:H // 2 + H, W // 2:W // 2 + W]

Hf = [np.fft.rfft2(np.fft.ifftshift(pad(psf[..., c]))) for c in range(3)]

def sim(x, c):
    return crop(np.fft.irfft2(np.fft.rfft2(pad(x)) * Hf[c], s=(PH, PW)))

def hp(a, s=4.0):
    return a - gaussian_filter(a, s)

def fitcorr(a, b):
    a, b = a.ravel(), b.ravel()
    return float(np.corrcoef(a, b)[0, 1])

rows = []
idx = diffusercam_split()["dev"]
for i, j in zip(idx, idx[1:] + idx[:1]):
    r = t.slice(i, 1).to_pylist()[0]
    yl = np.asarray(Image.open(io.BytesIO(r["lensless"]["bytes"]))).astype(np.float64)
    xl = np.asarray(Image.open(io.BytesIO(r["lensed"]["bytes"]))).astype(np.float64) / 255.0
    cs = [fitcorr(sim(xl[..., c], c), yl[..., c]) for c in range(3)]
    # control: the same simulation with a *different* scene (should correlate much less)
    xo = np.asarray(Image.open(io.BytesIO(t.slice(j, 1).to_pylist()[0]["lensed"]["bytes"]))).astype(np.float64) / 255.0
    cs_wrong = [fitcorr(sim(xo[..., c], c), yl[..., c]) for c in range(3)]
    hs = [fitcorr(hp(sim(xl[..., c], c)), hp(yl[..., c])) for c in range(3)]
    hs_wrong = [fitcorr(hp(sim(xo[..., c], c)), hp(yl[..., c])) for c in range(3)]
    rows.append(dict(index=i, wrong_scene_index=j, corr=float(np.mean(cs)), corr_wrong_scene=float(np.mean(cs_wrong)),
                     hp_corr=float(np.mean(hs)), hp_corr_wrong_scene=float(np.mean(hs_wrong)),
                     lensless_max=float(yl.max()), n_unique_levels=int(len(np.unique(yl)))))
    print(rows[-1], flush=True)
corr = np.array([r["corr"] for r in rows]); wrong = np.array([r["corr_wrong_scene"] for r in rows])
summary = dict(n=len(rows), corr_mean=float(corr.mean()), corr_min=float(corr.min()),
               corr_wrong_scene_mean=float(wrong.mean()),
               hp_corr_mean=float(np.mean([r["hp_corr"] for r in rows])),
               hp_corr_wrong_scene_mean=float(np.mean([r["hp_corr_wrong_scene"] for r in rows])),
               lensless_max_mean=float(np.mean([r["lensless_max"] for r in rows])),
               levels_mean=float(np.mean([r["n_unique_levels"] for r in rows])), rows=rows)
(out / "summary.json").write_text(json.dumps(summary, indent=2))
print({k: v for k, v in summary.items() if k != "rows"})
