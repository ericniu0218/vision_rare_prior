"""Diffusion Posterior Sampling (Chung et al., ICLR 2023), own implementation.

Matches the official code's update (guided_diffusion/condition_methods.py,
class PosteriorSampling, commit effbde7):

    out      = p_sample(x_t)                       # ancestral DDPM step -> x'_{t-1}, x0_hat
    norm     = || y - A(x0_hat(x_t)) ||_2          # un-squared norm
    x_{t-1}  = x'_{t-1} - scale * d norm / d x_t

Two deliberate, switchable differences from the official loop:
  * per_sample_norm=True computes one norm per batch element, so K posterior
    samples in a batch are independent. The official code takes a single norm
    over the whole batch, which couples samples when batch size > 1.
  * Randomness comes from an explicit torch.Generator. With
    ``official_rng=True`` and no generator, noise is drawn from the global RNG
    in the same order as the official loop (including the unused q_sample draw
    on the measurement), which lets tests reproduce official trajectories.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch

from ..diffusion.schedule import Schedule


@dataclass
class DPSConfig:
    scale: float = 0.3  # official Gaussian-deblur value; SR x4 uses 1.0 (check configs)
    per_sample_norm: bool = True
    official_rng: bool = False


def _randn_like(x, generator):
    if generator is None:
        return torch.randn_like(x)
    return torch.randn(x.shape, generator=generator, device=x.device, dtype=x.dtype)


def dps_step(model, sched: Schedule, op, y, x, t_idx: int, cfg: DPSConfig, generator=None,
             noise=None):
    """One DPS step at respaced index t_idx. Returns (x_{t-1}, x0_hat, residual_norm)."""
    t = torch.full((x.shape[0],), t_idx, device=x.device, dtype=torch.long)
    x = x.detach().requires_grad_(True)
    out = sched.p_mean_variance(model, x, t)
    if noise is None:
        noise = _randn_like(x, generator)
    sample = out["mean"]
    if t_idx != 0:
        sample = sample + torch.exp(0.5 * out["log_variance"]) * noise
    if cfg.official_rng and generator is None:
        torch.randn_like(y)  # the official loop draws q_sample noise here (unused by 'ps')
    r = y - op.A(out["pred_xstart"])
    if cfg.per_sample_norm:
        norms = r.reshape(r.shape[0], -1).abs().norm(dim=1)
        total = norms.sum()
    else:
        norms = torch.linalg.norm(r)[None]
        total = norms[0]
    (g,) = torch.autograd.grad(total, x)
    x_next = (sample - cfg.scale * g).detach()
    return x_next, out["pred_xstart"].detach(), norms.detach()


def dps_sample(model, sched: Schedule, op, y, cfg: DPSConfig = DPSConfig(), generator=None,
               x_T=None, callback=None):
    """Run the full reverse chain. y: measurement batch, one row per posterior sample.

    Returns the final image (the official code returns the last x, not x0_hat).
    """
    x = x_T if x_T is not None else _randn_like(_shape_like(y, op, model), generator)
    for t_idx in reversed(range(sched.T)):
        x, x0, norms = dps_step(model, sched, op, y, x, t_idx, cfg, generator)
        if callback is not None:
            callback(t_idx, x, x0, norms)
    return x


def _shape_like(y, op, model):
    """Image-shaped zeros for drawing x_T (image size from the model or the operator)."""
    if hasattr(op, "x_shape"):
        return torch.zeros((y.shape[0],) + tuple(op.x_shape), device=y.device)
    if hasattr(model, "in_channels") and hasattr(model, "image_size"):
        S = model.image_size
        return torch.zeros((y.shape[0], model.in_channels, S, S), device=y.device, dtype=y.dtype)
    return torch.zeros_like(y)  # image-shaped measurement (e.g. blur)
