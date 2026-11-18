"""Background-known-statistically (stationary Hotelling) detectability.

d'^2 = sum_f |H(f)|^2 |T(f)|^2 / (|H(f)|^2 P(f) + sigma^2)

where H is the operator's transfer function, T the (unit-amplitude) template's
spectrum and P the scene power spectrum (orthonormal FFT). Unlike the
background-known-exactly d' = ||A t|| / sigma, this counts natural-image
background variability as noise, which for small features after blur is 6-12x
larger than the measurement noise (FFHQ, blur std 3-5; see STATUS.md).
Valid for (approximately) shift-invariant operators: blur, and lensless with a
sensor-sized field of view (circular approximation on the padded grid).
"""
from __future__ import annotations

import numpy as np
import torch

from ..optics.psf import pad_center


def transfer_squared(op, shape) -> torch.Tensor:
    """|H(f)|^2 on the given grid for blur or lensless operators."""
    name = op.__class__.__name__
    if name == "GaussianBlur":
        return op.transfer_function(shape).abs() ** 2
    if name == "Lensless":
        h = op.psf.double().mean(0)
        hp = torch.fft.ifftshift(pad_center(h, shape), dim=(-2, -1))
        return torch.fft.fft2(hp).abs() ** 2
    raise NotImplementedError(f"no stationary transfer function for {name}")


def bks_dprime_unit(op, t: np.ndarray, P: torch.Tensor) -> float:
    """Hotelling d' of the template t at unit amplitude. P must be on the grid used for H."""
    shape = tuple(P.shape)
    H2 = transfer_squared(op, shape)
    tt = torch.from_numpy(t).double()
    T2 = torch.fft.fft2(pad_center(tt, shape), norm="ortho").abs() ** 2
    return float((H2 * T2 / (H2 * P + op.sigma ** 2)).sum().sqrt())


def estimate_power_spectrum(images: torch.Tensor, shape) -> torch.Tensor:
    """images: (N, H, W) luminance in [-1, 1]. Mean |FFT_ortho|^2 of zero-mean images on `shape`."""
    from ..optics.psf import scene_power_spectrum
    return scene_power_spectrum(images, shape)
