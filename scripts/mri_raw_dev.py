"""One engineering case from Stanford's public, legacy fully sampled 3D knee archive.

Run with case 1 only. The whole case is reserved for development. Raw acquisition
and reconstructions stay local. Output contains aggregate errors and site scores.
"""
from __future__ import annotations

import argparse
import json
import shutil
import tempfile
import zipfile
from pathlib import Path

import numpy as np
import sigpy as sp
import sigpy.mri as mr
import torch

from rareprior.diffusion.loader import sha256_file
from rareprior.features.insert import embed
from rareprior.features.library import template
from rareprior.metrics.observer import amplitude_ratio
from rareprior.operators.mri import Sense, poisson_mask, to_real2


def hybrid_slices(path, indices):
    """BART CFL is little-endian complex64 in Fortran order, with kx first."""
    with tempfile.TemporaryDirectory() as tmp, zipfile.ZipFile(path) as archive:
        prefix = "p1/e1/s1/kspace"
        hdr = archive.read(prefix + ".hdr").decode().splitlines()
        shape = tuple(int(x) for x in next(line for line in hdr if not line.startswith("#")).split())
        if len(shape) < 4 or np.prod(shape[4:]) != 1:
            raise ValueError(f"Expected one Cartesian multicoil volume, got {shape}")
        nx, ny, nz, nc = shape[:4]
        if not all(0 <= i < nx for i in indices):
            raise ValueError("Slice index outside the readout axis.")
        raw = Path(tmp) / "kspace.cfl"
        with archive.open(prefix + ".cfl") as source, raw.open("wb") as dest:
            shutil.copyfileobj(source, dest, length=8 << 20)
        if raw.stat().st_size != nx * ny * nz * nc * 8:
            raise ValueError("CFL byte count disagrees with its header.")
        volume = np.memmap(raw, dtype="<c8", mode="r", shape=(nx, ny, nz, nc), order="F")
        slices = np.empty((len(indices), nc, ny, nz), dtype=np.complex64)
        for c in range(nc):
            slices[:, c] = sp.ifft(volume[..., c], axes=(0,))[indices]
        return slices, shape


def run_slice(native, slice_id, accel, calib, support=None):
    mask = poisson_mask(native.shape[-2:], accel, calib, seed=42)
    cy, cx = (n // 2 for n in mask.shape)
    if not mask[cy - calib // 2:cy + calib // 2, cx - calib // 2:cx + calib // 2].all():
        raise ValueError("Undersampling mask does not preserve the ACS region.")
    masked = native * mask
    # EspiritCalib uses only the central calib × calib block of its input.
    maps = mr.app.EspiritCalib(masked, calib_width=calib, max_iter=50, show_pbar=False).run()
    finite = np.isfinite(maps).all()
    if not finite or not np.any(np.abs(maps) > 0):
        raise ValueError(f"Invalid ESPIRiT calibration on dev slice {slice_id}.")
    ref = mr.app.SenseRecon(native, maps, weights=np.ones_like(mask), lamda=1e-4,
                            max_iter=60, show_pbar=False).run()
    phase = torch.from_numpy(np.angle(ref))
    op = Sense(torch.from_numpy(maps), torch.from_numpy(mask))
    y_off = torch.from_numpy(masked[None])
    sp_a = mr.linop.Sense(maps, weights=mask.astype(np.float32))
    adj = sp_a.H(masked)
    ours = op.AT(y_off)
    adj2 = to_real2(torch.from_numpy(adj[None]))
    adj_error = float((ours - adj2).norm() / adj2.norm())

    # Fixed, separated sites chosen by a deterministic tissue-intensity proxy.
    mag = np.abs(ref)
    selected = []
    coords = np.unravel_index(np.argsort(mag, axis=None)[::-1], mag.shape)
    for j, (cy, cx) in enumerate(zip(*coords, strict=True)):
        if j == 0 and mag[cy, cx] == 0:
            raise ValueError("Empty reference image.")
        if 12 <= cy < mag.shape[0] - 12 and 12 <= cx < mag.shape[1] - 12:
            if all((cy - py) ** 2 + (cx - px) ** 2 >= 24 ** 2 for py, px in selected):
                selected.append((int(cy), int(cx)))
        if len(selected) == 2:
            break
    sites = [(y, x, cls) for (y, x), cls in zip(selected, ("blob", "line"), strict=True)]
    amplitude = 0.10
    injection = sum(embed(template(cls, 5), y, x, mag.shape, 2, phase) for y, x, cls in sites)
    y_on = masked + op.A(amplitude * injection)[0].numpy()

    def reconstruct(data, solver):
        if solver == "adjoint":
            return sp_a.H(data)
        cls = {"sense": mr.app.SenseRecon, "l1wavelet": mr.app.L1WaveletRecon}[solver]
        lam = {"sense": 1e-2, "l1wavelet": 2e-3}[solver]
        return cls(data, maps, weights=mask, lamda=lam, max_iter=60, show_pbar=False).run()

    if support is None:
        support = np.ones_like(mask)
    result = {"slice": slice_id, "target_accel": accel, "actual_accel": float(mask.size / mask.sum()),
              "native_support_accel": float(support.sum() / (mask & support).sum()),
              "acs_width": calib, "relative_adjoint_error": adj_error, "solver": {}}
    for solver in ("adjoint", "sense", "l1wavelet"):
        off = reconstruct(masked, solver)
        on = reconstruct(y_on, solver)
        diff = to_real2(torch.from_numpy((on - off)[None]))[0]
        result["solver"][solver] = {
            "mag_nrmse_vs_full_sense": float(np.linalg.norm(np.abs(off) - mag) / np.linalg.norm(mag)),
            "relative_sampled_residual": float(np.linalg.norm(sp_a(off) - masked) / np.linalg.norm(masked)),
            "paired_amplitude_response": [
                amplitude_ratio(diff, template(cls, 5), cy, cx, 1, amplitude, phase) for cy, cx, cls in sites],
        }
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--zip", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--padding-control", action="store_true",
                        help="Also undersample after zero-padding one development slice (labeled control).")
    parser.add_argument("--slices", type=int, nargs="+", default=[144, 160, 176])
    args = parser.parse_args()
    slices, shape = hybrid_slices(args.zip, args.slices)
    rows = []
    for slice_id, native in zip(args.slices, slices, strict=True):
        rss = sp.rss(sp.ifft(native, axes=(-2, -1)), axes=(0,))
        scale = float(np.quantile(rss, 0.99))
        if not np.isfinite(scale) or scale <= 0:
            raise ValueError(f"Invalid intensity scale on dev slice {slice_id}.")
        for accel in (4, 8):
            row = run_slice(native / scale, slice_id, accel, calib=24)
            row["rss_q99_scale"] = scale
            rows.append(row)
            print(slice_id, accel, row["relative_adjoint_error"], flush=True)
    padding = None
    if args.padding_control:
        native = slices[len(slices) // 2]
        borders = [(n // 4, n // 4) for n in native.shape[-2:]]
        padded = np.pad(native, [(0, 0), *borders])
        support = np.pad(np.ones(native.shape[-2:], dtype=bool), borders)
        rss = sp.rss(sp.ifft(padded, axes=(-2, -1)), axes=(0,))
        scale = float(np.quantile(rss, 0.99))
        padding = run_slice(padded / scale, args.slices[len(slices) // 2], 4, calib=24, support=support)
        padding["rss_q99_scale"] = scale
        padding["matrix_shape"] = list(support.shape)
        print("padding control", padding["actual_accel"], padding["native_support_accel"], flush=True)
    summary = {"archive": "Stanford legacy knee case 1, engineering development",
               "archive_sha256": sha256_file(args.zip), "header_shape": shape,
               "sigpy": sp.__version__, "torch": str(torch.__version__), "n_acquisitions": 1,
               "n_slices": len(slices), "rows": rows, "zero_padding_control": padding}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
