"""Multi-coil SENSE operator y = M F S x, plus undersampling masks.

Images are represented as real tensors with 2 channels (real, imaginary) so
that the same solvers and priors work for MRI and for natural images. k-space
is complex. FFTs are centred and orthonormal, matching SigPy's convention.
"""
from __future__ import annotations

import numpy as np
import torch

from .base import LinearOperator


def to_complex(x: torch.Tensor) -> torch.Tensor:
    """(B, 2, H, W) real -> (B, H, W) complex."""
    return torch.complex(x[:, 0], x[:, 1])


def to_real2(z: torch.Tensor) -> torch.Tensor:
    """(B, H, W) complex -> (B, 2, H, W) real."""
    return torch.stack([z.real, z.imag], dim=1)


def fft2c(x: torch.Tensor) -> torch.Tensor:
    x = torch.fft.ifftshift(x, dim=(-2, -1))
    x = torch.fft.fft2(x, norm="ortho")
    return torch.fft.fftshift(x, dim=(-2, -1))


def ifft2c(k: torch.Tensor) -> torch.Tensor:
    k = torch.fft.ifftshift(k, dim=(-2, -1))
    k = torch.fft.ifft2(k, norm="ortho")
    return torch.fft.fftshift(k, dim=(-2, -1))


class Sense(LinearOperator):
    """maps: (Ncoil, H, W) complex coil sensitivities. mask: (H, W) bool/0-1."""

    def __init__(self, maps: torch.Tensor, mask: torch.Tensor, sigma: float = 0.0):
        super().__init__()
        self.register_buffer("maps", maps.to(torch.complex128))
        self.register_buffer("mask", mask.to(torch.float64))
        self.sigma = sigma

    def A(self, x):
        z = to_complex(x).to(self.maps.dtype if x.dtype == torch.float64 else torch.complex64)
        maps = self.maps.to(z.dtype)
        k = fft2c(maps[None] * z[:, None])
        return k * self.mask.to(k.real.dtype)

    def add_noise(self, y, generator=None):
        """Noise only where k-space is sampled (unsampled entries stay exactly zero, as in real data;
        SigPy infers the sampling pattern from the nonzero entries)."""
        return super().add_noise(y, generator) * self.mask.to(y.real.dtype)

    def AT(self, y, x_shape=None):
        maps = self.maps.to(y.dtype)
        z = (maps.conj()[None] * ifft2c(y * self.mask.to(y.real.dtype))).sum(dim=1)
        return to_real2(z)


def equispaced_mask(shape, accel: int, calib: int = 24, axis: int = -1) -> np.ndarray:
    """1D Cartesian equispaced lines along ``axis`` plus a fully sampled centre band."""
    H, W = shape
    n = W if axis in (-1, 1) else H
    lines = np.zeros(n, dtype=bool)
    lines[::accel] = True
    c = n // 2
    lines[c - calib // 2:c + calib // 2] = True
    m = np.zeros(shape, dtype=bool)
    if axis in (-1, 1):
        m[:, lines] = True
    else:
        m[lines, :] = True
    return m


def variable_density_mask(shape, accel: float, calib: int = 24, power: float = 2.0,
                          seed: int = 0) -> np.ndarray:
    """2D variable-density random mask (fallback when SigPy is unavailable).

    Sampling probability falls as (1 - r)^power with radius r, scaled to hit
    the target acceleration, plus a fully sampled calib x calib centre.
    Prefer ``poisson_mask`` (SigPy's Poisson-disc) for experiments.
    """
    rng = np.random.default_rng(seed)
    H, W = shape
    yy, xx = np.meshgrid(np.linspace(-1, 1, H), np.linspace(-1, 1, W), indexing="ij")
    r = np.clip(np.sqrt(yy ** 2 + xx ** 2) / np.sqrt(2), 0, 1)
    p = (1 - r) ** power
    target = H * W / accel
    lo, hi = 0.0, 50.0
    for _ in range(60):  # bisection on the scale so that sum(p) ~= target
        s = (lo + hi) / 2
        if np.clip(s * p, 0, 1).sum() > target:
            hi = s
        else:
            lo = s
    m = rng.random(shape) < np.clip(lo * p, 0, 1)
    ch, cw = H // 2, W // 2
    m[ch - calib // 2:ch + calib // 2, cw - calib // 2:cw + calib // 2] = True
    return m


def poisson_mask(shape, accel: float, calib: int = 24, seed: int = 0) -> np.ndarray:
    """SigPy variable-density Poisson-disc mask (sigpy.mri.poisson)."""
    import sigpy.mri  # optional dependency
    m = sigpy.mri.poisson(shape, accel, calib=(calib, calib), seed=seed, dtype=np.complex64)
    return np.abs(m) > 0
