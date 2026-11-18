"""Real-checkpoint equivalence: own DPS vs the official DPS loop (same seed, same model).

Runs `--steps` respaced DPS steps on one FFHQ-sized input with both
implementations and reports the max abs difference. CPU-feasible for a few
steps; run the full chain on a GPU in Week 1.

  python scripts/check_official_equivalence.py --respacing 1000 --steps 3 --out results/equivalence
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import types
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from rareprior.diffusion.loader import download_checkpoint, load_adm, sha256_file  # noqa: E402
from rareprior.diffusion.schedule import Schedule, linear_betas  # noqa: E402
from rareprior.operators.blur import GaussianBlur  # noqa: E402
from rareprior.solvers.dps import DPSConfig, dps_step  # noqa: E402


def import_official(repo: Path):
    mb = types.ModuleType("motionblur")
    mbm = types.ModuleType("motionblur.motionblur")
    mbm.Kernel = object
    mb.motionblur = mbm
    sys.modules.setdefault("motionblur", mb)
    sys.modules.setdefault("motionblur.motionblur", mbm)
    sys.path.insert(0, str(repo))
    from guided_diffusion.condition_methods import get_conditioning_method
    from guided_diffusion.gaussian_diffusion import create_sampler
    from guided_diffusion.measurements import get_operator
    from guided_diffusion.unet import create_model
    return create_model, create_sampler, get_conditioning_method, get_operator


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default=str(ROOT / "third_party" / "dps"))
    ap.add_argument("--respacing", default="1000")
    ap.add_argument("--steps", type=int, default=5)
    ap.add_argument("--blur-backend", choices=["fft", "spatial"], default="fft")
    ap.add_argument("--out", default="results/equivalence")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(0)
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    ckpt = download_checkpoint("ffhq")
    digest = sha256_file(ckpt)
    ours_model = load_adm("ffhq", path=str(ckpt), device=dev)
    create_model, create_sampler, get_cond, get_operator = import_official(Path(args.repo))
    import yaml
    mcfg = yaml.safe_load((Path(args.repo) / "configs" / "model_config.yaml").read_text())
    mcfg["model_path"] = str(ckpt)
    off_model = create_model(**mcfg).to(dev).eval()

    # 1) identical weights and identical network outputs
    x = torch.randn(1, 3, 256, 256, device=dev)
    t = torch.tensor([500], device=dev)
    with torch.no_grad():
        e1, e2 = ours_model(x, t), off_model(x, t)
    net_diff = float((e1 - e2).abs().max())

    # 2) official operator vs ours (same kernel, reflect padding)
    off_op = get_operator("gaussian_blur", kernel_size=61, intensity=3.0, device=dev)
    op = GaussianBlur(61, 3.0, 0.05).to(dev)
    op.use_fft = args.blur_backend == "fft"
    with torch.no_grad():
        op_diff = float((off_op.forward(x) - op.A(x)).abs().max())

    # 3) DPS steps from the same state with the same noise draws
    class Noiser:
        __name__ = "gaussian"
    sampler = create_sampler("ddpm", 1000, "linear", "epsilon", "learned_range", False, True, False,
                             timestep_respacing="" if args.respacing == "1000" else args.respacing)
    cond = get_cond("ps", off_op, Noiser(), scale=0.3)
    sched = Schedule(linear_betas(1000), respacing=args.respacing)
    y = op.A(torch.rand(1, 3, 256, 256, device=dev) * 2 - 1) + 0.05 * torch.randn(1, 3, 256, 256, device=dev)
    x_off = torch.randn(1, 3, 256, 256, device=dev)
    x_own = x_off.clone()
    diffs, local_diffs, times = [], [], []
    for k in range(args.steps):
        idx = sched.T - 1 - k
        seed = 1000 + k
        torch.manual_seed(seed)
        t0 = time.time()
        time_t = torch.tensor([idx], device=dev)
        shared_state = x_off.detach()
        img = shared_state.requires_grad_()
        o = sampler.p_sample(x=img, t=time_t, model=off_model)
        noisy = sampler.q_sample(y, t=time_t)
        x_off, _ = cond.conditioning(x_t=o["sample"], measurement=y, noisy_measurement=noisy, x_prev=img,
                                     x_0_hat=o["pred_xstart"])
        x_off = x_off.detach()
        t_off = time.time() - t0
        torch.manual_seed(seed)
        t0 = time.time()
        x_own, _, _ = dps_step(ours_model, sched, op, y, x_own, idx, DPSConfig(scale=0.3, official_rng=True))
        times.append(time.time() - t0)
        torch.manual_seed(seed)
        local, _, _ = dps_step(ours_model, sched, op, y, shared_state, idx, DPSConfig(scale=0.3, official_rng=True))
        local_diffs.append(float((x_off - local).abs().max()))
        diffs.append(float((x_off - x_own).abs().max()))
        print(f"step {k} (t={idx}): max|diff| = {diffs[-1]:.2e}  official {t_off:.1f}s, ours {times[-1]:.1f}s",
              flush=True)
    res = {"device": dev, "blur_backend": args.blur_backend,
           "checkpoint_sha256": digest, "network_output_max_diff": net_diff,
           "operator_max_diff": op_diff, "respacing": args.respacing, "step_max_diffs": diffs,
           "same_input_step_max_diffs": local_diffs, "cumulative_pass_1e_5": max(diffs) < 1e-5,
           "own_seconds_per_step": float(np.mean(times)), "torch": torch.__version__}
    (out / "summary.json").write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
