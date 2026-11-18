"""Compare the generalized DiffPIR data step with the upstream circular FFT step."""
from __future__ import annotations

import argparse
import importlib.util
import json
import subprocess
from pathlib import Path

import torch

from rareprior.diffusion.schedule import Schedule, linear_betas
from rareprior.operators.blur import GaussianBlur
from rareprior.solvers.cg import cg_normal
from rareprior.solvers.diffpir import circular_proximal

UPSTREAM = "2a9898129a1b274131b98746e5b364bc20adc1e1"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    rev = subprocess.check_output(["git", "-C", str(args.repo), "rev-parse", "HEAD"], text=True).strip()
    if rev != UPSTREAM:
        raise ValueError(f"Use the pinned upstream revision {UPSTREAM}.")
    spec = importlib.util.spec_from_file_location("diffpir_fft", args.repo / "utils/utils_sisr.py")
    ref = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ref)
    op = GaussianBlur(31, 3.0, .05, boundary="circular")
    g = torch.Generator().manual_seed(71)
    image = torch.rand(1, 3, 64, 64, dtype=torch.float64, generator=g) * 2 - 1
    y = op.add_noise(op.A(image), generator=g)
    x0 = op.A(image)
    k = op.kernel[None, None].double()
    y01 = (1 + y) / 2
    fft = ref.pre_calculate(y01, k, 1)
    schedule = Schedule(linear_betas(1000))
    rows = []
    for t in (999, 600, 250, 50, 1):
        a = schedule.acp[t]
        rho = (op.sigma / 2) ** 2 * a / (1 - a)
        x_ref = 2 * ref.data_solution((1 + x0) / 2, *fft, rho, 1) - 1
        b_data = op.AT(y)
        b = b_data + rho * x0
        h2 = op.transfer_function(y.shape[-2:]).abs().square()
        direct = circular_proximal(x0, torch.fft.fft2(b_data), h2, rho)
        direct_rms = float((direct - x_ref).square().mean().sqrt())
        for iters in (20, 100, 500):
            x = cg_normal(op, b, rho, x0=x0, iters=iters, tol=1e-10)
            normal_residual = (op.normal(x) + rho * x - b).norm() / b.norm()
            rows.append(dict(timestep=t, rho=float(rho), iterations_limit=iters,
                             relative_normal_residual=float(normal_residual), direct_fft_rms_difference=direct_rms,
                             rms_difference=float((x - x_ref).square().mean().sqrt())))
    summary = dict(kind="synthetic circular proximal subproblem only", reference=rev,
                   torch=str(torch.__version__), rows=rows)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
