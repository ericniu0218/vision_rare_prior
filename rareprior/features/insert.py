"""Insert features with matched measurement-domain detectability d'.

For a known template t at a known location and Gaussian measurement noise,
the ideal (background-known-exactly) observer's detectability is

    d' = ||A t||_2 / sigma_eff,

with sigma_eff = sigma for real noise and sigma / sqrt(2) for circular complex
noise of total variance sigma^2 (MRI). The amplitude a of each feature is set
so that d'(a t) equals the target, so every feature carries the same
information into the measurement. Differences in survival are then caused by
the reconstruction, not by the camera.
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import numpy as np
import torch

from .library import CLASSES, N_VARIANTS, SIZES, template


@dataclass
class Site:
    kind: str          # "feature" or "control"
    cy: int
    cx: int
    cls: str = ""
    size: int = 0
    variant: int = 0
    sign: int = 1
    amplitude: float = 0.0
    dprime: float = 0.0
    texture: float = 0.0   # local background std (32x32 window)
    level: float = 0.0     # design level: target d' ("dprime" mode) or amplitude ("fixed" mode)
    dprime_bke: float = 0.0  # background-known-exactly d' = a ||A t|| / sigma, always recorded

    def to_dict(self):
        return asdict(self)


def embed(t: np.ndarray, cy: int, cx: int, shape, channels: int, phase: torch.Tensor | None = None,
          dtype=torch.float32) -> torch.Tensor:
    """Place template t centred at (cy, cx) in a (1, channels, H, W) zero image.

    For 2-channel complex images, ``phase`` (H, W) gives the background phase;
    the feature is t * exp(i * phase) so it is consistent with the local signal.
    """
    H, W = shape
    r = t.shape[0] // 2
    out = torch.zeros((1, channels, H, W), dtype=dtype)
    tt = torch.from_numpy(t).to(dtype)
    y0, y1, x0, x1 = cy - r, cy + r + 1, cx - r, cx + r + 1
    if y0 < 0 or x0 < 0 or y1 > H or x1 > W:
        raise ValueError("template out of bounds")
    if phase is not None:
        ph = phase[y0:y1, x0:x1].to(dtype)
        out[0, 0, y0:y1, x0:x1] = tt * torch.cos(ph)
        out[0, 1, y0:y1, x0:x1] = tt * torch.sin(ph)
    else:
        out[0, :, y0:y1, x0:x1] = tt
    return out


def sigma_eff(op) -> float:
    is_complex = getattr(op, "complex_measurement", False) or op.__class__.__name__ == "Sense"
    return op.sigma / math.sqrt(2) if is_complex else op.sigma


def measurement_norm(op, t_full: torch.Tensor) -> float:
    with torch.no_grad():
        return float(op.A(t_full.double() if _op_is_double(op) else t_full).abs().norm())


def _op_is_double(op) -> bool:
    for b in op.buffers():
        return b.dtype in (torch.float64, torch.complex128)
    return False


def amplitude_for_dprime(op, t_full: torch.Tensor, dprime: float) -> float:
    return dprime * sigma_eff(op) / measurement_norm(op, t_full)


def ideal_tpr(dprime: float, fpr: float = 0.05) -> float:
    """TPR of the ideal observer at a given FPR: 1 - Phi(z_{1-fpr} - d')."""
    from scipy.stats import norm
    return float(1 - norm.cdf(norm.ppf(1 - fpr) - dprime))


@dataclass
class InsertConfig:
    n_features: int = 6
    n_controls: int = 8
    margin: int = 32            # features live in [margin, H - margin)
    min_sep: int = 48           # feature-to-feature distance (avoid interaction)
    ctrl_sep: int = 24          # control-to-any-site distance
    dprimes: tuple = (2.0, 4.0)
    sizes: tuple | None = None  # None = all sizes in the library
    classes: tuple = CLASSES
    value_range: tuple | None = (-1.0, 1.0)   # None for MRI
    max_tries: int = 2000
    # "dprime": amplitude set so every feature has the same measurement d' (comparing reconstructions
    # under one camera). "fixed": amplitudes drawn from `amplitudes` in image units (comparing cameras).
    # controls pass the same pixel-range acceptance test as a random hypothetical feature, so
    # feature and control sites come from the same location distribution (dev pilot: without
    # this, the U-Net's counterfactual FPR was 0.089 against a 0.05 target)
    match_controls: bool = True
    amplitude_mode: str = "dprime"
    amplitudes: tuple = (0.2, 0.4)
    # which d' the "dprime" mode matches: "bke" = ||A t|| / sigma (background known exactly) or
    # "bks" = stationary Hotelling d' with scene power spectrum P (background known statistically).
    dprime_kind: str = "bke"
    power_spectrum: str | None = None   # path to a saved P(f) tensor (torch.save), needed for "bks"


def _local_std(x: torch.Tensor, cy: int, cx: int, w: int = 16) -> float:
    H, W = x.shape[-2:]
    patch = x[..., max(0, cy - w):min(H, cy + w), max(0, cx - w):min(W, cx + w)]
    return float(patch.std())


_P_CACHE: dict = {}
_BKS_CACHE: dict = {}


def _op_key(op, ps_path) -> tuple:
    """Cache key from the operator's parameters (never id(op), which Python can reuse)."""
    if hasattr(op, "psf"):
        p = op.psf.double()
        moment = float((p * torch.arange(p.numel()).view_as(p)).sum())
        sig = (tuple(p.shape), float(p.sum()), float((p ** 2).sum()), moment)
    else:
        sig = (getattr(op, "kernel_size", None), getattr(op, "std", None), getattr(op, "boundary", None))
    return (op.__class__.__name__, float(op.sigma), sig, ps_path)


def _power_spectrum(path):
    if path not in _P_CACHE:
        _P_CACHE[path] = torch.load(path, weights_only=True).double()
    return _P_CACHE[path]


def sample_sites(x: torch.Tensor, op, rng: np.random.Generator, cfg: InsertConfig,
                 phase: torch.Tensor | None = None, assignments=None):
    """Choose feature and control sites for one image x of shape (C, H, W).

    assignments: optional list of (cls, size, dprime) for the features (from a
    balanced design); otherwise drawn uniformly.
    Returns (x_with_features, sites, dropped) where dropped counts features
    skipped because they could not fit in the value range.
    """
    C, H, W = x.shape
    sites: list[Site] = []
    centres: list[tuple[int, int]] = []
    feat = torch.zeros((1, C, H, W), dtype=x.dtype)
    dropped = 0

    def far_enough(cy, cx, sep=cfg.min_sep):
        return all((cy - a) ** 2 + (cx - b) ** 2 >= sep ** 2 for a, b in centres)

    for i in range(cfg.n_features):
        if assignments is not None:
            cls, size, dp = assignments[i]
        else:
            cls = cfg.classes[rng.integers(len(cfg.classes))]
            sizes = [s for s in SIZES[cls] if cfg.sizes is None or s in cfg.sizes]
            size = sizes[rng.integers(len(sizes))]
            dp = (cfg.dprimes if cfg.amplitude_mode == "dprime" else cfg.amplitudes)[
                rng.integers(len(cfg.dprimes if cfg.amplitude_mode == "dprime" else cfg.amplitudes))]
        variant = int(rng.integers(N_VARIANTS[cls]))
        sign = int(rng.choice([-1, 1]))
        t = template(cls, size, variant)
        placed = False
        m = max(cfg.margin, t.shape[0] // 2 + 1)  # the template must fit inside the image
        for _ in range(cfg.max_tries // 20):
            cy = int(rng.integers(m, H - m))
            cx = int(rng.integers(m, W - m))
            if not far_enough(cy, cx):
                continue
            t_full = embed(t, cy, cx, (H, W), C, phase=phase, dtype=x.dtype)
            if cfg.amplitude_mode == "dprime" and cfg.dprime_kind == "bks":
                key = (_op_key(op, cfg.power_spectrum), cls, size, variant)
                if key not in _BKS_CACHE:
                    from .hotelling import bks_dprime_unit
                    _BKS_CACHE[key] = bks_dprime_unit(op, t, _power_spectrum(cfg.power_spectrum))
                a = dp / _BKS_CACHE[key]
                d_here = dp
            elif cfg.amplitude_mode == "dprime":
                a = amplitude_for_dprime(op, t_full, dp)
                d_here = dp
            else:  # fixed amplitude; record the d' this camera actually gives the feature
                a = float(dp)
                d_here = a * measurement_norm(op, t_full) / max(sigma_eff(op), 1e-12)
            cand = x[None] + feat + sign * a * t_full
            if cfg.value_range is not None:
                lo, hi = cfg.value_range
                if cand.min() < lo or cand.max() > hi:
                    continue
            feat = feat + sign * a * t_full
            centres.append((cy, cx))
            d_bke = a * measurement_norm(op, t_full) / max(sigma_eff(op), 1e-12)
            sites.append(Site("feature", cy, cx, cls, size, variant, sign, a, float(d_here),
                              _local_std(x, cy, cx), level=float(dp), dprime_bke=float(d_bke)))
            placed = True
            break
        if not placed:
            dropped += 1
    levels = cfg.dprimes if cfg.amplitude_mode == "dprime" else cfg.amplitudes
    for _ in range(cfg.n_controls):
        # hypothetical feature that the control site must be able to hold
        hcls = cfg.classes[rng.integers(len(cfg.classes))]
        hsizes = [s for s in SIZES[hcls] if cfg.sizes is None or s in cfg.sizes]
        hsize = hsizes[rng.integers(len(hsizes))]
        hvar = int(rng.integers(N_VARIANTS[hcls]))
        hlev = levels[rng.integers(len(levels))]
        hsign = int(rng.choice([-1, 1]))
        ht = template(hcls, hsize, hvar)
        m = max(cfg.margin, ht.shape[0] // 2 + 1)
        for _ in range(cfg.max_tries):
            cy = int(rng.integers(m, H - m))
            cx = int(rng.integers(m, W - m))
            if not far_enough(cy, cx, cfg.ctrl_sep):
                continue
            if cfg.match_controls and cfg.value_range is not None:
                t_full = embed(ht, cy, cx, (H, W), C, phase=phase, dtype=x.dtype)
                if cfg.amplitude_mode == "dprime" and cfg.dprime_kind == "bks":
                    key = (_op_key(op, cfg.power_spectrum), hcls, hsize, hvar)
                    if key not in _BKS_CACHE:
                        from .hotelling import bks_dprime_unit
                        _BKS_CACHE[key] = bks_dprime_unit(op, ht, _power_spectrum(cfg.power_spectrum))
                    a = hlev / _BKS_CACHE[key]
                elif cfg.amplitude_mode == "dprime":
                    a = amplitude_for_dprime(op, t_full, hlev)
                else:
                    a = float(hlev)
                cand = x[None] + feat + hsign * a * t_full
                lo, hi = cfg.value_range
                if cand.min() < lo or cand.max() > hi:
                    continue
            centres.append((cy, cx))
            sites.append(Site("control", cy, cx, texture=_local_std(x, cy, cx)))
            break
    return (x[None] + feat)[0], sites, dropped
