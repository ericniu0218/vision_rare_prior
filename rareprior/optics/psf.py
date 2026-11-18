"""PSF parameterisations, design objectives and optimisation (H4).

All objectives use a stationary Gaussian model on the padded grid (circular
convolution, scene power spectrum P(f), white read noise sigma^2). They are
cheap and differentiable. Optimised PSFs must always be *evaluated* with the
full pad-and-crop Lensless operator and real reconstructors.

Objectives (orthonormal FFT for scenes and noise; H = unnormalised FFT of h):
  mse     minimise  sum_f P sigma^2 / (|H|^2 P + sigma^2)          (Wiener MSE)
  mi      maximise  1/2 sum_f log(1 + |H|^2 P / sigma^2)           (Gaussian MI)
  detect  maximise  mean_t sum_f |H|^2 |T_t|^2 / (|H|^2 P + sigma^2)
          = average Hotelling d'^2 of the feature library under a random
            Gaussian background (Barrett & Myers).
Equal total light: h >= 0 and sum(h) = 1 (so H(0) = 1).
"""
from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn.functional as F

from ..features.library import all_cells, template


# ---------------------------------------------------------------- parameterisations
class FreeFormPSF(torch.nn.Module):
    def __init__(self, shape, init: torch.Tensor | None = None):
        super().__init__()
        if init is None:
            init = torch.rand(shape, dtype=torch.float64)
        init = init.double().clamp_min(1e-12)
        init = init / init.sum()
        # inverse softplus of a scaled init so optimisation starts at `init`
        scaled = init * init.numel()
        self.w = torch.nn.Parameter(torch.log(torch.expm1(scaled.clamp_min(1e-6))))

    def forward(self):
        h = F.softplus(self.w)
        return h / h.sum()


class PhaseMaskPSF(torch.nn.Module):
    """Intensity PSF |ASM_z(exp(i phi))|^2 from a smooth phase mask.

    phi is a coarse grid upsampled bicubically; the angular spectrum method
    propagates the field a distance z. Non-negativity and energy conservation
    hold by construction (we also renormalise to sum 1 over the sensor).
    """

    def __init__(self, shape, coarse: int = 32, wavelength: float = 550e-9, dx: float = 4e-6,
                 z: float = 2e-3, phase_scale: float = 20.0, seed: int = 0):
        super().__init__()
        g = torch.Generator().manual_seed(seed)
        self.shape = shape
        self.phi_c = torch.nn.Parameter(phase_scale * torch.randn((1, 1, coarse, coarse), generator=g,
                                                                  dtype=torch.float64))
        H, W = shape
        fy = torch.fft.fftfreq(H, d=dx, dtype=torch.float64)
        fx = torch.fft.fftfreq(W, d=dx, dtype=torch.float64)
        FY, FX = torch.meshgrid(fy, fx, indexing="ij")
        arg = (1 / wavelength) ** 2 - FX ** 2 - FY ** 2
        prop = torch.where(arg > 0, torch.exp(1j * 2 * math.pi * z * torch.sqrt(arg.clamp_min(0))),
                           torch.zeros_like(arg, dtype=torch.complex128))
        self.register_buffer("transfer", prop)

    def phase(self):
        return F.interpolate(self.phi_c, size=self.shape, mode="bicubic", align_corners=False)[0, 0]

    def forward(self):
        u0 = torch.exp(1j * self.phase())
        uz = torch.fft.ifft2(torch.fft.fft2(u0) * self.transfer)
        h = uz.abs() ** 2
        return h / h.sum()


# ---------------------------------------------------------------- spectra
def pad_center(x: torch.Tensor, shape):
    H, W = x.shape[-2:]
    PH, PW = shape
    t, l = (PH - H) // 2, (PW - W) // 2
    return F.pad(x, (l, PW - W - l, t, PH - H - t))


def transfer(h: torch.Tensor, pad_shape) -> torch.Tensor:
    hp = torch.fft.ifftshift(pad_center(h, pad_shape), dim=(-2, -1))
    return torch.fft.fft2(hp)


def scene_power_spectrum(scenes: torch.Tensor, pad_shape) -> torch.Tensor:
    """scenes: (N, H, W) grayscale in [-1, 1]. Mean |FFT_ortho|^2 of zero-mean padded scenes."""
    s = scenes.double()
    s = s - s.mean(dim=(-2, -1), keepdim=True)
    X = torch.fft.fft2(pad_center(s, pad_shape), norm="ortho")
    return (X.abs() ** 2).mean(0)


def template_spectra(pad_shape, sizes=(3, 5)) -> torch.Tensor:
    """|T(f)|^2 for every small template variant, unit energy, centred on the grid."""
    out = []
    for cls, size in all_cells(sizes):
        from ..features.library import N_VARIANTS
        for v in range(N_VARIANTS[cls]):
            t = torch.from_numpy(template(cls, size, v))
            t = t / t.norm()
            T = torch.fft.fft2(pad_center(t, pad_shape), norm="ortho")
            out.append(T.abs() ** 2)
    return torch.stack(out)


# ---------------------------------------------------------------- objectives (all "lower is better")
def loss_mse(H2, P, sigma2):
    return (P * sigma2 / (H2 * P + sigma2)).sum()


def loss_mi(H2, P, sigma2):
    return -0.5 * torch.log1p(H2 * P / sigma2).sum()


def loss_detect(H2, P, sigma2, T2):
    return -(T2 * (H2 / (H2 * P + sigma2))[None]).sum(dim=(-2, -1)).mean()


def objective(name, h, pad_shape, P, sigma2, T2=None):
    H2 = transfer(h, pad_shape).abs() ** 2
    if name == "mse":
        return loss_mse(H2, P, sigma2)
    if name == "mi":
        return loss_mi(H2, P, sigma2)
    if name == "detect":
        return loss_detect(H2, P, sigma2, T2)
    raise ValueError(name)


def optimize_psf(param: torch.nn.Module, name: str, pad_shape, P, sigma2, T2=None,
                 steps: int = 2000, lr: float = 0.05, log_every: int = 200):
    opt = torch.optim.Adam(param.parameters(), lr=lr)
    hist = []
    for k in range(steps):
        opt.zero_grad()
        loss = objective(name, param(), pad_shape, P, sigma2, T2)
        loss.backward()
        opt.step()
        if k % log_every == 0 or k == steps - 1:
            hist.append((k, float(loss)))
    with torch.no_grad():
        return param().detach(), hist


# ---------------------------------------------------------------- multiplexing
def participation_ratio(h: torch.Tensor) -> float:
    """(sum h)^2 / (N sum h^2): 1/N for a single bright pixel, 1 for a uniform PSF."""
    h = h.double().flatten()
    return float(h.sum() ** 2 / (h.numel() * (h ** 2).sum()))


def n_peaks(h: torch.Tensor, rel: float = 0.1) -> int:
    """Number of strict local maxima above rel * max (3x3 neighbourhood)."""
    a = h.double()[None, None]
    mx = F.max_pool2d(a, 3, stride=1, padding=1)
    peaks = (a == mx) & (a > rel * a.max())
    return int(peaks.sum())


def summarize(h: torch.Tensor, pad_shape, P, sigma2, T2) -> dict:
    with torch.no_grad():
        return {
            "mse": float(objective("mse", h, pad_shape, P, sigma2)),
            "mi_nats": float(-objective("mi", h, pad_shape, P, sigma2)),
            "detect_d2": float(-objective("detect", h, pad_shape, P, sigma2, T2)),
            "participation_ratio": participation_ratio(h),
            "n_peaks": n_peaks(h),
        }


def as_numpy_image(h: torch.Tensor) -> np.ndarray:
    a = h.detach().double().cpu().numpy()
    return a / a.max()
