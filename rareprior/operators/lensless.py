"""Lensless (DiffuserCam-style) forward model: y = C(h * P(x)) + n.

P zero-pads the scene to twice its size, * is linear (non-circular) convolution
implemented with FFTs on the padded grid, and C crops the centre back to the
sensor size. Antipa et al., Optica 2018; Monakhova et al., Optics Express 2019.

Simplification (documented in the plan): the scene support equals the sensor
size. In the full DiffuserCam model the unknown lives on the padded grid and
light from outside the sensor's field of view also reaches the sensor.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

from .base import LinearOperator


def _pad_center(x: torch.Tensor, shape) -> torch.Tensor:
    H, W = x.shape[-2:]
    PH, PW = shape
    top, left = (PH - H) // 2, (PW - W) // 2
    return F.pad(x, (left, PW - W - left, top, PH - H - top))


def _crop_center(x: torch.Tensor, shape) -> torch.Tensor:
    PH, PW = x.shape[-2:]
    H, W = shape
    top, left = (PH - H) // 2, (PW - W) // 2
    return x[..., top:top + H, left:left + W]


class Lensless(LinearOperator):
    """psf: tensor (C, H, W) or (1, H, W), non-negative. Normalised to sum 1 per channel."""

    def __init__(self, psf: torch.Tensor, sigma: float = 0.01, normalize: bool = True, fov: str = "sensor"):
        """fov="sensor": scene support = sensor size (simplified model).
        fov="full": scene lives on the 2x padded grid, as in the real DiffuserCam model; light from
        outside the sensor's footprint still reaches the sensor (and a lens would miss it)."""
        super().__init__()
        assert fov in ("sensor", "full")
        self.fov = fov
        if psf.ndim == 2:
            psf = psf[None]
        psf = psf.double().clamp_min(0)
        if normalize:
            psf = psf / psf.sum(dim=(-2, -1), keepdim=True)
        self.register_buffer("psf", psf)
        self.sigma = sigma
        self.sensor_shape = tuple(psf.shape[-2:])
        self.pad_shape = (2 * self.sensor_shape[0], 2 * self.sensor_shape[1])
        self.x_shape_hw = self.pad_shape if fov == "full" else self.sensor_shape

    def _H(self, dtype, device):
        h = _pad_center(self.psf, self.pad_shape)
        h = torch.fft.ifftshift(h, dim=(-2, -1))
        return torch.fft.rfft2(h.to(device=device, dtype=dtype))

    def A(self, x):
        Hf = self._H(x.dtype, x.device)
        xp = x if self.fov == "full" else _pad_center(x, self.pad_shape)
        y = torch.fft.irfft2(torch.fft.rfft2(xp) * Hf, s=self.pad_shape)
        return _crop_center(y, self.sensor_shape)

    def AT(self, y, x_shape=None):
        Hf = self._H(y.dtype, y.device)
        yp = _pad_center(y, self.pad_shape)
        x = torch.fft.irfft2(torch.fft.rfft2(yp) * Hf.conj(), s=self.pad_shape)
        return x if self.fov == "full" else _crop_center(x, self.sensor_shape)


def prepare_psf(psf: np.ndarray, out_shape=(256, 256), square_crop: bool = True,
                bg_quantile: float = 0.05) -> torch.Tensor:
    """Background-subtract, optionally centre-crop to a square, resize, normalise.

    psf: (H, W) or (H, W, C) array from psf.tiff. Returns (C, h, w) float64.
    The same crop and resize must be applied to real measurements (see
    ``prepare_measurement``) so that the model stays consistent.
    """
    a = np.asarray(psf, dtype=np.float64)
    if a.ndim == 2:
        a = a[..., None]
    a = a - np.quantile(a, bg_quantile, axis=(0, 1), keepdims=True)
    a = np.clip(a, 0, None)
    t = torch.from_numpy(a).permute(2, 0, 1)[None]
    if square_crop:
        s = min(t.shape[-2:])
        t = _crop_center(t, (s, s))
    t = F.interpolate(t, size=out_shape, mode="area")[0]
    return t / t.sum(dim=(-2, -1), keepdim=True)


def prepare_measurement(y: np.ndarray, out_shape=(256, 256), square_crop: bool = True) -> torch.Tensor:
    """Apply the same geometric transform as ``prepare_psf`` to a real capture. Returns (C, h, w)."""
    a = np.asarray(y, dtype=np.float64)
    if a.ndim == 2:
        a = a[..., None]
    t = torch.from_numpy(a).permute(2, 0, 1)[None]
    if square_crop:
        s = min(t.shape[-2:])
        t = _crop_center(t, (s, s))
    return F.interpolate(t, size=out_shape, mode="area")[0]


def lenslet_psf(shape, k: int, spot_sigma: float = 0.7, seed: int = 0, margin: float = 0.1) -> torch.Tensor:
    """Multiplexing family (after Kabuli et al., Optica 2026): k equal-weight Gaussian spots at random
    positions inside the sensor-sized PSF window. k = 1 is a lens (one spot at the centre)."""
    H, W = shape
    g = torch.Generator().manual_seed(seed)
    yy, xx = torch.meshgrid(torch.arange(H, dtype=torch.float64), torch.arange(W, dtype=torch.float64),
                            indexing="ij")
    if k == 1:
        cy, cx = torch.tensor([H / 2]), torch.tensor([W / 2])
    else:
        cy = margin * H + (1 - 2 * margin) * H * torch.rand(k, generator=g, dtype=torch.float64)
        cx = margin * W + (1 - 2 * margin) * W * torch.rand(k, generator=g, dtype=torch.float64)
    h = torch.zeros(shape, dtype=torch.float64)
    for a, b in zip(cy.tolist(), cx.tolist()):
        h += torch.exp(-((yy - a) ** 2 + (xx - b) ** 2) / (2 * spot_sigma ** 2))
    return (h / h.sum())[None]
