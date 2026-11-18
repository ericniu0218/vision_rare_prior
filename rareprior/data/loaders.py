"""Data loading: FFHQ (HF mirror), DiffuserCam (HF copy of the Waller Lab dataset).

All images are returned as float32 tensors (C, H, W) in [-1, 1].
"""
from __future__ import annotations

import io
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

DIFFUSERCAM_REPO = "bezzam/DiffuserCam-Lensless-Mirflickr-Dataset"
DIFFUSERCAM_REVISION = "957cb8e28dc9bf610521495f1874d579cc894257"
FFHQ_REPO = "merkol/ffhq-256"
FFHQ_REVISION = "f23c0e21f6fe222b4429272a86be00e4583b0298"


def to_tensor(img) -> torch.Tensor:
    """PIL image or HxW(xC) array (uint8 or float) -> (C, H, W) float32 in [-1, 1]."""
    a = np.asarray(img)
    if a.dtype == np.uint8:
        a = a.astype(np.float32) / 255.0
    elif a.dtype == np.uint16:
        a = a.astype(np.float32) / 65535.0
    else:
        a = a.astype(np.float32)
        if a.max() > 1.0:
            a = a / a.max()
    if a.ndim == 2:
        a = a[..., None]
    return torch.from_numpy(a).permute(2, 0, 1) * 2 - 1


def center_square_resize(x: torch.Tensor, size: int = 256) -> torch.Tensor:
    C, H, W = x.shape
    s = min(H, W)
    t, l = (H - s) // 2, (W - s) // 2
    x = x[:, t:t + s, l:l + s]
    return F.interpolate(x[None], size=(size, size), mode="area")[0]




def _ffhq_shard_files(repo, revision):
    from huggingface_hub import HfApi
    files = HfApi().list_repo_files(repo, repo_type="dataset", revision=revision)
    return sorted(f for f in files if f.endswith(".parquet"))


def load_ffhq(indices, repo: str = FFHQ_REPO, size: int = 256, revision: str | None = None):
    """Load FFHQ images by original index (matched on the stored filename, e.g. '69001.png').

    Downloads only the shards that contain the requested indices (~490 MB each).
    """
    import pyarrow.parquet as pq
    from huggingface_hub import hf_hub_download
    from PIL import Image
    want = {int(i) for i in indices}
    revision = revision or (FFHQ_REVISION if repo == FFHQ_REPO else None)
    if not revision:
        raise ValueError("A custom FFHQ mirror requires an explicit revision.")
    files = _ffhq_shard_files(repo, revision)
    per = 70000 / len(files)
    shards = sorted({min(len(files) - 1, int(i // per)) for i in want})
    out = {}
    tried = set()
    while shards:
        k = shards.pop(0)
        if k in tried or not 0 <= k < len(files):
            continue
        tried.add(k)
        t = pq.read_table(hf_hub_download(repo, files[k], repo_type="dataset", revision=revision))
        col = t.column("image").to_pylist()
        names = [int(Path(c["path"]).stem) for c in col]
        for name, c in zip(names, col):
            if name in want:
                img = Image.open(io.BytesIO(c["bytes"])).convert("RGB")
                out[name] = center_square_resize(to_tensor(img), size)
        missing = want - set(out)
        if missing:  # boundary rounding: look at neighbours
            lo, hi = min(names), max(names)
            if any(m < lo for m in missing):
                shards.append(k - 1)
            if any(m > hi for m in missing):
                shards.append(k + 1)
    if want - set(out):
        raise KeyError(f"indices not found: {sorted(want - set(out))[:5]}")
    return out


def load_diffusercam_test(indices=None, cache_dir=None):
    """Rows of the HF DiffuserCam test split: dict idx -> (lensless, lensed) arrays."""
    from huggingface_hub import hf_hub_download
    import pyarrow.parquet as pq
    from PIL import Image
    path = hf_hub_download(DIFFUSERCAM_REPO, "data/test-00000-of-00001.parquet", repo_type="dataset",
                           cache_dir=cache_dir, revision=DIFFUSERCAM_REVISION)
    table = pq.read_table(path)
    n = table.num_rows
    indices = range(n) if indices is None else indices
    out = {}
    for i in indices:
        row = table.slice(int(i), 1).to_pylist()[0]
        lensless = np.asarray(Image.open(io.BytesIO(row["lensless"]["bytes"])))
        lensed = np.asarray(Image.open(io.BytesIO(row["lensed"]["bytes"])))
        out[int(i)] = (lensless, lensed)
    return out


def load_diffusercam_psf(cache_dir=None) -> np.ndarray:
    from huggingface_hub import hf_hub_download
    path = hf_hub_download(DIFFUSERCAM_REPO, "psf.tiff", repo_type="dataset", cache_dir=cache_dir,
                           revision=DIFFUSERCAM_REVISION)
    try:
        import tifffile
        return tifffile.imread(path)
    except ImportError:
        from PIL import Image
        return np.asarray(Image.open(path))
