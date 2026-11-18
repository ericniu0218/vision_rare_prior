"""Week-0 environment check. Run on Kaggle (GPU) and locally (CPU).

Prints PASS / FAIL / SKIP for each item of the plan's Week-0 checklist that
can be automated, and writes verify_env.json. Never aborts on the first failure.

  python scripts/verify_env.py [--full]     # --full downloads the 2.2 GB ImageNet checkpoint too
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

RESULTS = {}


def check(name):
    def deco(fn):
        def run(*a, **k):
            t0 = time.time()
            try:
                info = fn(*a, **k)
                RESULTS[name] = {"status": "PASS", "info": info, "s": round(time.time() - t0, 1)}
            except _Skip as e:
                RESULTS[name] = {"status": "SKIP", "info": str(e)}
            except Exception as e:  # noqa: BLE001
                RESULTS[name] = {"status": "FAIL", "info": f"{type(e).__name__}: {e}",
                                 "trace": traceback.format_exc(limit=3)}
            print(f"[{RESULTS[name]['status']}] {name}: {RESULTS[name]['info']}", flush=True)
        return run
    return deco


class _Skip(Exception):
    pass


@check("torch/gpu")
def c_torch():
    import torch
    return {"torch": torch.__version__, "cuda": torch.cuda.is_available(),
            "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"}


@check("ffhq checkpoint (HF mirror, SHA-256, strict load)")
def c_ffhq():
    from rareprior.diffusion.loader import load_adm
    m = load_adm("ffhq")
    return {"params_M": round(sum(p.numel() for p in m.parameters()) / 1e6, 1)}


@check("ffhq unconditional sample is face-like (saved to verify_env_ffhq_sample.png)")
def c_ffhq_sample():
    import torch

    from rareprior.diffusion.loader import load_adm
    from rareprior.diffusion.schedule import Schedule, linear_betas
    from rareprior.solvers.dps import DPSConfig, dps_sample
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    m = load_adm("ffhq", device=dev)
    steps = 100 if dev == "cuda" else 25

    class Zero:
        sigma = 0.0

        def A(self, x):
            return x * 0
    s = Schedule(linear_betas(1000), respacing=str(steps))
    g = torch.Generator(device=dev).manual_seed(0)
    x = dps_sample(m, s, Zero(), torch.zeros(1, 3, 256, 256, device=dev), DPSConfig(scale=0.0), generator=g)
    _save(x, "verify_env_ffhq_sample.png")
    return {"steps": steps, "range": [float(x.min()), float(x.max())]}


@check("imagenet checkpoint (2.2 GB)")
def c_imagenet(full):
    if not full:
        raise _Skip("pass --full")
    from rareprior.diffusion.loader import load_adm
    m = load_adm("imagenet")
    return {"params_M": round(sum(p.numel() for p in m.parameters()) / 1e6, 1)}


@check("DiffuserCam HF test split + PSF")
def c_diffusercam():
    from rareprior.data.loaders import load_diffusercam_psf, load_diffusercam_test
    from rareprior.data.splits import diffusercam_split
    i = diffusercam_split()["dev"][0]
    psf = load_diffusercam_psf()
    lensless, lensed = load_diffusercam_test([i])[i]
    return {"psf_shape": list(psf.shape), "psf_dtype": str(psf.dtype),
            "lensless_shape": list(lensless.shape), "lensed_shape": list(lensed.shape)}


@check("SigPy (poisson mask, ESPIRiT, Sense, L1-wavelet)")
def c_sigpy():
    import numpy as np
    import sigpy as sp
    import sigpy.mri as mr
    mask = mr.poisson((64, 64), 4, calib=(16, 16), seed=0, dtype=np.complex64)
    mps = mr.birdcage_maps((4, 64, 64))
    img = sp.shepp_logan((64, 64))
    ksp = mr.linop.Sense(mps)(img)
    maps = mr.app.EspiritCalib(ksp, calib_width=16, show_pbar=False).run()
    rec = mr.app.L1WaveletRecon(ksp * (np.abs(mask) > 0), maps, lamda=1e-3, show_pbar=False).run()
    return {"sigpy": sp.__version__, "accel": float(mask.size / (np.abs(mask) > 0).sum()),
            "recon_nrmse": float(np.linalg.norm(np.abs(rec) - np.abs(img)) / np.linalg.norm(img))}


@check("official DPS repo importable (third_party/dps)")
def c_official():
    repo = Path(__file__).resolve().parents[1] / "third_party" / "dps"
    if not (repo / "guided_diffusion").exists():
        raise _Skip("git submodule update --init (or clone DPS2022/diffusion-posterior-sampling there)")
    return {"path": str(repo)}


@check("DPS speed on this device (s/step, 256px, batch 1)")
def c_speed():
    import torch

    from rareprior.diffusion.loader import load_adm
    from rareprior.diffusion.schedule import Schedule, linear_betas
    from rareprior.operators.blur import GaussianBlur
    from rareprior.solvers.dps import DPSConfig, dps_step
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    m = load_adm("ffhq", device=dev)
    op = GaussianBlur().to(dev)
    s = Schedule(linear_betas(1000))
    x = torch.randn(1, 3, 256, 256, device=dev)
    y = torch.zeros_like(x)
    n = 10 if dev == "cuda" else 2
    dps_step(m, s, op, y, x, 500, DPSConfig())
    if dev == "cuda":
        torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(n):
        x, _, _ = dps_step(m, s, op, y, x, 500, DPSConfig())
    if dev == "cuda":
        torch.cuda.synchronize()
    per = (time.time() - t0) / n
    return {"s_per_step": round(per, 3), "s_per_1000_step_sample": round(per * 1000, 1),
            "peak_mem_GB": round(torch.cuda.max_memory_allocated() / 1e9, 2) if dev == "cuda" else None}


def _save(x, path):
    import numpy as np
    from PIL import Image
    a = ((x[0].detach().cpu().clamp(-1, 1) + 1) * 127.5).permute(1, 2, 0).numpy().astype(np.uint8)
    Image.fromarray(a).save(path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--full", action="store_true")
    ap.add_argument("--skip-speed", action="store_true")
    args = ap.parse_args()
    c_torch()
    c_ffhq()
    c_ffhq_sample()
    c_imagenet(args.full)
    c_diffusercam()
    c_sigpy()
    c_official()
    if not args.skip_speed:
        c_speed()
    Path("verify_env.json").write_text(json.dumps(RESULTS, indent=2, default=str))
    bad = [k for k, v in RESULTS.items() if v["status"] == "FAIL"]
    print("\nFAILED:" if bad else "\nall automated checks passed or skipped", bad if bad else "")


if __name__ == "__main__":
    main()
