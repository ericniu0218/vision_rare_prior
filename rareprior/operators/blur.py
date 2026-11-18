"""Gaussian blur, built to match the official DPS operator exactly.

Official DPS (util/img_utils.py, class Blurkernel): the kernel is
scipy.ndimage.gaussian_filter applied to a centred delta of size kernel_size,
the image is reflection-padded by kernel_size // 2, then depthwise conv2d.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from scipy import ndimage

from .base import LinearOperator


def dps_gaussian_kernel(kernel_size: int = 61, std: float = 3.0) -> np.ndarray:
    n = np.zeros((kernel_size, kernel_size))
    n[kernel_size // 2, kernel_size // 2] = 1
    return ndimage.gaussian_filter(n, sigma=std)


class GaussianBlur(LinearOperator):
    def __init__(self, kernel_size: int = 61, std: float = 3.0, sigma: float = 0.05, channels: int = 3,
                 boundary: str = "reflect"):
        """boundary: "reflect" (official DPS) or "circular" (official DiffPIR uses mode='wrap')."""
        super().__init__()
        assert boundary in ("reflect", "circular")
        self.boundary = boundary
        self.kernel_size = kernel_size
        self.std = std
        self.sigma = sigma
        self.channels = channels
        k = torch.from_numpy(dps_gaussian_kernel(kernel_size, std))
        self.register_buffer("kernel", k.float())

    use_fft: bool = True  # same result as conv2d to float precision; much faster for 61x61 kernels

    def _conv_valid(self, xp: torch.Tensor) -> torch.Tensor:
        """'valid' correlation of padded input with the (symmetric) kernel."""
        k = self.kernel.to(dtype=xp.dtype, device=xp.device)
        ks = self.kernel_size
        if not self.use_fft:
            c = xp.shape[1]
            return F.conv2d(xp, k.expand(c, 1, -1, -1), groups=c)
        Hp, Wp = xp.shape[-2:]
        s = (Hp, Wp)  # circular conv of size (Hp, Wp); the valid region has no wrap-around
        K = torch.fft.rfft2(torch.flip(k, (0, 1)), s=s)
        full = torch.fft.irfft2(torch.fft.rfft2(xp, s=s) * K, s=s)
        return full[..., ks - 1:, ks - 1:]

    def A(self, x):
        p = self.kernel_size // 2
        return self._conv_valid(F.pad(x, (p, p, p, p), mode=self.boundary))

    def AT(self, y, x_shape=None):
        # Reflection padding makes the closed-form adjoint fiddly; autograd is exact.
        from .base import autograd_adjoint
        return autograd_adjoint(self.A, y, y.shape if x_shape is None else x_shape)

    def transfer_function(self, shape, dtype=torch.float64):
        """|H(f)|^2 under a circular approximation (used only for analytic checks)."""
        h = torch.zeros(shape[-2:], dtype=dtype, device=self.kernel.device)
        k = self.kernel.to(dtype)
        ks = self.kernel_size
        h[:ks, :ks] = k
        h = torch.roll(h, shifts=(-(ks // 2), -(ks // 2)), dims=(0, 1))
        return torch.fft.fft2(h)
