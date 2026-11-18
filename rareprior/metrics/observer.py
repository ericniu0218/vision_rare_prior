"""Matched-filter observer, per-solver thresholds and survival.

Score at a site (background-known-exactly matched filter):
    s = <x_hat - x_clean, w> / ||w||   over the template window, w = signed template with its
    mean removed (DC_REMOVE), so smooth background errors do not look like features.
For control sites (no feature) the score is computed for every template in
the library and both signs, which gives the null distribution per cell.

Threshold tau for a cell = the (1 - fpr) quantile of that cell's null scores,
computed separately for each solver, so each solver is judged at its own 5%
false-alarm rate. Survival = share of inserted features with s > tau.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import torch

from ..features.library import N_VARIANTS, all_cells, template


def _patch(img: torch.Tensor, cy: int, cx: int, r: int) -> torch.Tensor:
    """(2r+1)x(2r+1) window centred at (cy, cx); zero-padded where it leaves the image."""
    H, W = img.shape[-2:]
    if cy - r >= 0 and cx - r >= 0 and cy + r < H and cx + r < W:
        return img[..., cy - r:cy + r + 1, cx - r:cx + r + 1]
    padded = torch.nn.functional.pad(img, (r, r, r, r))
    return padded[..., cy:cy + 2 * r + 1, cx:cx + 2 * r + 1]


DC_REMOVE = True  # zero-mean templates: smooth background errors do not look like features


def _weights(t: np.ndarray, channels: int, phase_patch=None, dc_remove: bool | None = None) -> torch.Tensor:
    tt = torch.from_numpy(t)
    if phase_patch is not None:
        w = torch.stack([tt * torch.cos(phase_patch), tt * torch.sin(phase_patch)])
    else:
        w = tt.expand(channels, -1, -1).clone()
    if DC_REMOVE if dc_remove is None else dc_remove:
        w = w - w.mean(dim=(-2, -1), keepdim=True)
    return w


def site_score(diff: torch.Tensor, t: np.ndarray, cy: int, cx: int, sign: int = 1,
               phase: torch.Tensor | None = None) -> float:
    """diff: (C, H, W) = x_hat - x_clean."""
    r = t.shape[0] // 2
    p = _patch(diff.double(), cy, cx, r)
    ph = None if phase is None else _patch(phase.double(), cy, cx, r)
    w = _weights(t, diff.shape[0], ph).double()
    return float(sign * (p * w).sum() / w.norm())


def amplitude_ratio(diff, t, cy, cx, sign, amplitude, phase=None) -> float:
    """<x_hat - x_clean, t> / <a t, t>: 1 = fully recovered, 0 = erased."""
    r = t.shape[0] // 2
    p = _patch(diff.double(), cy, cx, r)
    ph = None if phase is None else _patch(phase.double(), cy, cx, r)
    w = _weights(t, diff.shape[0], ph, dc_remove=False).double()
    return float(sign * (p * w).sum() / (amplitude * (w * w).sum()))


def score_image(x_hat: torch.Tensor, x_clean: torch.Tensor, sites, meta: dict,
                cells=None, phase=None) -> list[dict]:
    """Score every site in one reconstruction. Returns rows for a DataFrame.

    meta: identifiers copied into every row (image_id, solver, severity, sample...).
    Feature rows: one per feature. Control rows: one per (cell, variant, sign).
    """
    diff = (x_hat - x_clean).detach().cpu()
    if diff.ndim == 4:
        diff = diff[0]
    cells = cells or all_cells()
    rows = []
    for s in sites:
        if s.kind == "feature":
            t = template(s.cls, s.size, s.variant)
            rows.append(dict(meta, kind="feature", cls=s.cls, size=s.size, variant=s.variant, sign=s.sign,
                             dprime=s.dprime, level=s.level, dprime_bke=s.dprime_bke,
                             amplitude=s.amplitude, texture=s.texture, cy=s.cy, cx=s.cx,
                             score=site_score(diff, t, s.cy, s.cx, s.sign, phase),
                             ratio=amplitude_ratio(diff, t, s.cy, s.cx, s.sign, s.amplitude, phase)))
        else:
            for cls, size in cells:
                for v in range(N_VARIANTS[cls]):
                    t = template(cls, size, v)
                    base = site_score(diff, t, s.cy, s.cx, 1, phase)
                    for sign in (-1, 1):
                        rows.append(dict(meta, kind="control", cls=cls, size=size, variant=v, sign=sign,
                                         texture=s.texture, cy=s.cy, cx=s.cx, score=sign * base))
    return rows


GROUP = ["solver", "severity", "cls", "size"]


def thresholds(df: pd.DataFrame, fpr: float = 0.05, group=GROUP) -> pd.DataFrame:
    ctrl = df[df.kind == "control"]
    return ctrl.groupby(group)["score"].quantile(1 - fpr).rename("tau").reset_index()


def add_detection(df: pd.DataFrame, fpr: float = 0.05, group=GROUP, taus: pd.DataFrame | None = None):
    """Adds column 'detected' to feature rows using per-cell control thresholds."""
    taus = thresholds(df, fpr, group) if taus is None else taus
    feats = df[df.kind == "feature"].merge(taus, on=group, how="left")
    if feats["tau"].isna().any():
        raise ValueError("some feature cells have no control scores")
    feats["detected"] = (feats["score"] > feats["tau"]).astype(int)
    return feats


def empirical_fpr(df: pd.DataFrame, taus: pd.DataFrame, group=GROUP) -> pd.DataFrame:
    """FPR achieved on a (different) set of null scores, e.g. counterfactual nulls."""
    ctrl = df[df.kind == "control"].merge(taus, on=group)
    ctrl["fp"] = (ctrl["score"] > ctrl["tau"]).astype(int)
    return ctrl.groupby(group)["fp"].mean().reset_index()


def survival_table(feats: pd.DataFrame, by=("solver", "severity", "cls", "size", "level")) -> pd.DataFrame:
    from ..features.insert import ideal_tpr
    f = feats.copy()
    if "level" not in f:
        f["level"] = f["dprime"]
    f["ideal"] = f["dprime"].map(ideal_tpr)
    return f.groupby(list(by)).agg(survival=("detected", "mean"), n=("detected", "size"),
                                   ratio=("ratio", "median"), ideal_tpr=("ideal", "mean"),
                                   dprime_mean=("dprime", "mean")).reset_index()
