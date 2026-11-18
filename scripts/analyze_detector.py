"""H3: can we locate erased features without ground truth?

Positives: feature sites erased in the majority of posterior samples (detected in
< 50% of samples). Negatives: features that survived, and control sites.
Scores (computed by run_survival, never using x_clean): det_residual, det_semantic,
det_variance. Reports AUROC per score and the paired, image-clustered bootstrap
of AUROC(residual) - AUROC(variance) and AUROC(semantic) - AUROC(variance).

  python scripts/analyze_detector.py --runs artifacts/survival/<name>/<hash> --out results/<name>_detector
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rareprior.metrics.observer import add_detection  # noqa: E402
from analyze_survival import load_runs  # noqa: E402

SCORES = ["det_residual", "det_semantic", "det_pixel_pair", "det_variance"]


def auroc(scores: np.ndarray, labels: np.ndarray) -> float:
    """Mann-Whitney AUROC with average ranks for ties."""
    s = pd.Series(scores).rank(method="average").to_numpy()
    pos = labels == 1
    n1, n0 = pos.sum(), (~pos).sum()
    if n1 == 0 or n0 == 0:
        return float("nan")
    return float((s[pos].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def average_precision(scores, labels) -> float:
    """Average precision at distinct score thresholds, with ties grouped."""
    order = np.argsort(-scores)
    labels = labels[order]
    endpoints = np.r_[np.flatnonzero(np.diff(scores[order])), len(scores) - 1]
    tp = np.cumsum(labels)[endpoints]
    return float(np.sum(np.diff(tp, prepend=0) * tp / (endpoints + 1)) / tp[-1])


def site_table(df: pd.DataFrame) -> pd.DataFrame:
    feats = add_detection(df[df.kind.isin(["feature", "control"])])
    f = feats.groupby(["image_id", "solver", "severity", "cy", "cx"]).agg(
        frac_detected=("detected", "mean"), **{c: (c, "first") for c in SCORES if c in feats}).reset_index()
    f["label"] = (f.frac_detected < 0.5).astype(int)
    c = df[df.kind == "control"].groupby(["image_id", "solver", "severity", "cy", "cx"]).agg(
        **{s: (s, "first") for s in SCORES if s in df}).reset_index()
    c["label"] = 0
    return pd.concat([f, c], ignore_index=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-boot", type=int, default=5000)
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    df, _ = load_runs(args.runs)
    t = site_table(df)
    res = {}
    rng = np.random.default_rng(0)
    for (solver, severity), g in t.groupby(["solver", "severity"]):
        avail = [s for s in SCORES if s in g and g[s].notna().all()]
        positives = int(g.label.sum())
        r = {"n_images": int(g.image_id.nunique()), "n_sites": int(len(g)), "n_erased": positives,
             "auprc": {s: average_precision(g[s].to_numpy(), g.label.to_numpy()) for s in avail} if positives else {},
             "auroc": {s: auroc(g[s].to_numpy(), g.label.to_numpy()) for s in avail}}
        imgs = g.image_id.unique()
        by_img = {i: g[g.image_id == i] for i in imgs}
        for s in [x for x in avail if x != "det_variance" and "det_variance" in avail]:
            deltas = []
            for _ in range(args.n_boot):
                b = pd.concat([by_img[i] for i in rng.choice(imgs, len(imgs))])
                deltas.append(auroc(b[s].to_numpy(), b.label.to_numpy()) -
                              auroc(b["det_variance"].to_numpy(), b.label.to_numpy()))
            d = np.array(deltas)
            d = d[~np.isnan(d)]
            r[f"{s}_minus_variance"] = {"delta": r["auroc"][s] - r["auroc"]["det_variance"],
                                        "ci_low": float(np.quantile(d, 0.025)) if len(d) else None,
                                        "ci_high": float(np.quantile(d, 0.975)) if len(d) else None}
        res[f"{solver}/{severity}"] = r
    (out / "summary.json").write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
