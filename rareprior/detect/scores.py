"""Erasure detectors that use only the posterior samples and the measurement (H3).

  residual  : energy of the back-projected data residual A^T(y - A x_hat) in the
              site's window, divided by its expected value under noise alone.
              Erased features that the camera did record leave a residual.
  semantic  : mean pairwise LPIPS distance between the K samples' crops
              (or mean pairwise squared pixel distance when explicitly selected).
  variance  : per-pixel variance across the K samples, averaged over the window
              (the baseline).
"""
from __future__ import annotations

import itertools

import torch


def window(img, cy, cx, r):
    return img[..., cy - r:cy + r + 1, cx - r:cx + r + 1]


def noise_only_residual_energy(op, x_shape, cy, cx, r, n_draws: int = 16, seed: int = 0) -> float:
    """E || [A^T n]_window ||^2 under the operator's noise model (Monte Carlo)."""
    g = torch.Generator().manual_seed(seed)
    zeros = torch.zeros(x_shape, dtype=torch.float64)
    y0 = op.A(zeros)
    tot = 0.0
    for _ in range(n_draws):
        n = op.add_noise(torch.zeros_like(y0), generator=g)
        tot += float((window(op.AT(n, x_shape=x_shape), cy, cx, r) ** 2).sum())
    return tot / n_draws


def residual_score(op, y, x_hat, cy, cx, r, null_energy: float) -> float:
    res = op.AT(y - op.A(x_hat), x_shape=x_hat.shape)
    return float((window(res, cy, cx, r) ** 2).sum()) / max(null_energy, 1e-30)


def variance_score(samples: torch.Tensor, cy, cx, r) -> float:
    """samples: (K, C, H, W)."""
    return float(window(samples, cy, cx, r).var(dim=0, unbiased=True).mean())


def semantic_score(samples: torch.Tensor, cy, cx, crop: int = 32, use_lpips: bool = True) -> float:
    crops = window(samples, cy, cx, crop)
    pairs = list(itertools.combinations(range(samples.shape[0]), 2))
    if use_lpips:
        from ..metrics.image_quality import lpips_distance
        a = torch.stack([crops[i] for i, _ in pairs])
        b = torch.stack([crops[j] for _, j in pairs])
        if a.shape[1] != 3:
            raise ValueError("LPIPS requires RGB crops.")
        return float(lpips_distance(a, b).mean())
    return float(torch.stack([(crops[i] - crops[j]).pow(2).mean() for i, j in pairs]).mean())
