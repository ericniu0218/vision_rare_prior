"""Image-clustered paired bootstrap for survival differences (H1, H4).

Thresholds are re-estimated inside every bootstrap replicate (weighted
quantiles of control scores), so the CI includes threshold uncertainty.
Pooled survival gives equal weight to every (severity, cls, size, dprime) cell.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

TAU_KEYS = ["severity", "cls", "size"]
CELL_KEYS = ["severity", "cls", "size", "level"]


def _weighted_quantile(sorted_scores, sorted_w, q):
    if not len(sorted_scores):
        return np.nan
    cum = np.cumsum(sorted_w)
    total = cum[-1]
    if total <= 0:
        return np.nan
    rank = q * (total - 1)
    low = np.floor(rank)
    high = np.ceil(rank)
    a, b = sorted_scores[np.searchsorted(cum, [low, high], side="right")]
    return a + (b - a) * (rank - low)


@dataclass
class _Cell:
    key: tuple
    ctrl_s: np.ndarray
    ctrl_i: np.ndarray
    feats: dict  # dprime -> (scores, img_idx)


def _prepare(df: pd.DataFrame, solver: str, img_index: dict, cell_filter=None, tau_keys=TAU_KEYS):
    d = df[df.solver == solver]
    if cell_filter is not None:
        d = d[cell_filter(d)]
    cells = []
    ctrl = d[d.kind == "control"]
    feat = d[d.kind == "feature"]
    for key, f in feat.groupby(list(tau_keys)):
        c = ctrl
        for k, v in zip(tau_keys, key):
            c = c[c[k] == v]
        order = np.argsort(c["score"].to_numpy())
        cs = c["score"].to_numpy()[order]
        ci = c["image_id"].map(img_index).to_numpy()[order]
        lvl = "level" if "level" in f else "dprime"
        fd = {dp: (g["score"].to_numpy(), g["image_id"].map(img_index).to_numpy())
              for dp, g in f.groupby(lvl)}
        cells.append(_Cell(key, cs, ci, fd))
    return cells


def _pooled_survival(cells, w, fpr, pooling):
    vals, counts = [], []
    for cell in cells:
        tau = _weighted_quantile(cell.ctrl_s, w[cell.ctrl_i], 1 - fpr)
        for s, i in cell.feats.values():
            wi = w[i]
            count = wi.sum()
            if count <= 0:
                if pooling == "macro":
                    return np.nan  # changing the supported cells changes the estimand
                continue
            if not np.isfinite(tau):
                return np.nan
            vals.append((wi * (s > tau)).sum() / count)
            counts.append(count)
    return float(np.average(vals, weights=counts if pooling == "site" else None)) if vals else np.nan


def paired_survival_bootstrap(df: pd.DataFrame, solver_a: str, solver_b: str, n_boot: int = 10000,
                              fpr: float = 0.05, cell_filter=None, seed: int = 0, alpha: float = 0.05,
                              tau_keys=TAU_KEYS, pooling="macro"):
    """Delta = S_a - S_b with a percentile CI from an image-clustered paired bootstrap.

    df: rows from metrics.observer.score_image with columns image_id, solver,
    severity, kind, cls, size, dprime, score. Images must be shared by both solvers.
    """
    if pooling not in {"macro", "site"}:
        raise ValueError(f"Unknown pooling mode: {pooling}")
    imgs = sorted(set(df[df.solver == solver_a].image_id) & set(df[df.solver == solver_b].image_id))
    if not imgs:
        raise ValueError("no shared images")
    idx = {im: k for k, im in enumerate(imgs)}
    d = df[df.image_id.isin(imgs)]
    ca, cb = (_prepare(d, solver_a, idx, cell_filter, tau_keys), _prepare(d, solver_b, idx, cell_filter, tau_keys))
    n = len(imgs)
    w1 = np.ones(n)
    sa, sb = _pooled_survival(ca, w1, fpr, pooling), _pooled_survival(cb, w1, fpr, pooling)
    if not np.isfinite([sa, sb]).all():
        raise ValueError("The observed contrast has missing feature or null support.")
    rng = np.random.default_rng(seed)
    deltas = np.empty(n_boot)
    for b in range(n_boot):
        w = np.bincount(rng.integers(0, n, n), minlength=n).astype(float)
        deltas[b] = _pooled_survival(ca, w, fpr, pooling) - _pooled_survival(cb, w, fpr, pooling)
    valid = deltas[np.isfinite(deltas)]
    interval_supported = len(valid) >= 0.95 * n_boot
    lo, hi = np.quantile(valid, [alpha / 2, 1 - alpha / 2]) if interval_supported else (None, None)
    return {"S_a": sa, "S_b": sb, "delta": sa - sb, "ci_low": lo, "ci_high": hi,
            "n_images": n, "n_boot": n_boot, "n_boot_valid": len(valid), "pooling": pooling,
            "boot_sd": float(valid.std()) if interval_supported else None,
            "p_value": bootstrap_pvalue(valid) if interval_supported else None}


def paired_metric_bootstrap(a: np.ndarray, b: np.ndarray, n_boot: int = 10000, seed: int = 0, alpha=0.05):
    """Per-image paired difference of means (e.g. LPIPS), percentile CI."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    rng = np.random.default_rng(seed)
    n = len(a)
    idx = rng.integers(0, n, (n_boot, n))
    deltas = (a[idx] - b[idx]).mean(1)
    lo, hi = np.quantile(deltas, [alpha / 2, 1 - alpha / 2])
    return {"delta": float((a - b).mean()), "ci_low": float(lo), "ci_high": float(hi)}


def holm(pvalues: dict, alpha: float = 0.05) -> dict:
    """Holm step-down. Returns {name: (p, reject)}."""
    items = sorted(pvalues.items(), key=lambda kv: kv[1])
    m = len(items)
    out, stop = {}, False
    for k, (name, p) in enumerate(items):
        thresh = alpha / (m - k)
        reject = (not stop) and p <= thresh
        if not reject:
            stop = True
        out[name] = (p, reject)
    return out


def bootstrap_pvalue(deltas: np.ndarray) -> float:
    """Two-sided bootstrap p-value for H0: delta = 0 (share of replicates on the far side, x2)."""
    deltas = np.asarray(deltas)
    p = 2 * (1 + min((deltas <= 0).sum(), (deltas >= 0).sum())) / (1 + len(deltas))
    return float(min(1.0, p))
