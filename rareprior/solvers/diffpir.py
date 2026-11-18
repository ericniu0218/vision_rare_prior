"""DiffPIR (Zhu et al., CVPRW 2023), own implementation for general linear operators.

Follows the official main_ddpir_deblur.py (commit 2a98981), "pred_xstart" mode:
  * timesteps on a quadratic grid of NFE points, from T-1 down to ~0;
  * at each step: x0 = Tweedie estimate (clipped), then the data subproblem
        x0 <- argmin ||y - A x||^2 + rho_t ||x - x0||^2,
    with rho_t = lambda * sigma_01^2 / sigma_t^2 and sigma_t = sqrt(1-abar)/sqrt(abar);
  * then x_{t-1} = sqrt(abar_{t-1}) x0 + sqrt(1-abar_{t-1}) (sqrt(1-zeta) eps_hat + sqrt(zeta) n).

The official demo uses circular blur, a noisy-measurement start, and an FFT
proximal update in [0, 1] units. This variant uses a Gaussian start, an FFT
step for circular blur and truncated CG otherwise. In [-1, 1] units the same
subproblem uses rho_t = lambda * max(sigma/2, 0.001)^2 / sigma_t^2. The demo's
default noise is 0.10 in [-1, 1], whereas the shared FFHQ experiment uses 0.05.
The whole reconstruction is a variant, even when the proximal steps agree.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from ..diffusion.schedule import Schedule
from ..operators.blur import GaussianBlur
from .cg import cg_normal


@dataclass
class DiffPIRConfig:
    lam: float = 1.0
    zeta: float = 0.1
    nfe: int = 100
    cg_iters: int = 20
    skip_type: str = "quad"


def diffpir_timesteps(T: int, nfe: int, skip_type: str = "quad") -> list[int]:
    """Descending list of (original) timesteps visited, as in the official code."""
    if skip_type == "quad":
        seq = np.sqrt(np.linspace(0, T ** 2, nfe))
        seq = [int(s) for s in seq]
        seq[-1] = seq[-1] - 1
    else:
        skip = T // nfe
        seq = [i * skip for i in range(nfe)]
        if skip > 1:
            seq.append(T - 1)
    # official: sigma index i -> timestep T-1-i (reduced_alpha_cumprod is monotone)
    return [T - 1 - s for s in seq]


def circular_proximal(x0, rhs_fft, transfer_sq, rho):
    """Exact regularized least-squares step for circular convolution."""
    return torch.fft.ifft2((rhs_fft + rho * torch.fft.fft2(x0)) / (transfer_sq + rho)).real


def diffpir_sample(model, sched: Schedule, op, y, sigma: float, cfg: DiffPIRConfig = DiffPIRConfig(),
                   generator=None, x_shape=None):
    assert sched.T == len(sched.timestep_map) and sched.timestep_map == list(range(sched.T)), \
        "DiffPIR uses the full (un-respaced) schedule; it picks its own timesteps"
    acp = torch.from_numpy(sched.acp)
    ts = diffpir_timesteps(sched.T, cfg.nfe, cfg.skip_type)
    if x_shape is None:  # image and measurement have the same shape (blur); pass it otherwise
        x_shape = y.shape

    def randn():
        return torch.randn(x_shape, generator=generator, device=y.device, dtype=y.real.dtype)

    x = randn()
    b_data = op.AT(y, x_shape=x_shape)
    circular = isinstance(op, GaussianBlur) and op.boundary == "circular"
    if circular:
        h2 = op.transfer_function(x_shape[-2:], dtype=y.dtype).abs().square()
        b_fft = torch.fft.fft2(b_data)
    x0 = None
    for i, t in enumerate(ts):
        tt = torch.full((x.shape[0],), t, device=x.device, dtype=torch.long)
        with torch.no_grad():
            eps, _ = sched.eps_and_var(model, x, tt)
            x0 = sched.pred_xstart(x, tt, eps)
        if i == len(ts) - 1:
            break
        a_t = float(acp[t])
        sigma_t2 = (1 - a_t) / a_t
        rho = cfg.lam * max(sigma / 2, 1e-3) ** 2 / sigma_t2
        if circular:
            x0 = circular_proximal(x0, b_fft, h2, rho)
        else:
            x0 = cg_normal(op, b_data + rho * x0, rho, x0=x0, iters=cfg.cg_iters)
        t_next = ts[i + 1]
        a_n = float(acp[t_next])
        eps_hat = (x - np.sqrt(a_t) * x0) / np.sqrt(1 - a_t)
        x = (np.sqrt(a_n) * x0 + np.sqrt(1 - cfg.zeta) * np.sqrt(1 - a_n) * eps_hat
             + np.sqrt(cfg.zeta) * np.sqrt(1 - a_n) * randn())
    return x0
