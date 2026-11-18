"""Check same-site null calibration on disjoint FFHQ development images.

Feature-on and feature-off reconstructions share site, noise and sampler seed.
Known-background and reconstruction-only observers are reported separately.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from analyze_survival import load_runs

KEYS = ["image_id", "solver", "severity", "sample", "cy", "cx", "cls", "size", "variant", "sign"]


def assess(rows, n_fit):
    features = rows[rows.kind == "feature"]
    null = rows[rows.kind == "cf_null"]
    if features.empty or null.empty:
        raise ValueError("Run with counterfactual=true first.")
    cols = ["score", "no_clean_score"]
    if "measurement_score" in features and features.measurement_score.notna().all():
        cols.append("measurement_score")
    matched = features[KEYS + cols].merge(null[KEYS + cols], on=KEYS, how="outer",
                                         suffixes=("_on", "_off"), validate="one_to_one", indicator=True)
    if not matched["_merge"].eq("both").all():
        raise ValueError("Same-site feature and null rows must pair exactly.")
    ids = sorted(matched.image_id.unique())
    if not 0 < n_fit < len(ids):
        raise ValueError("Calibration and evaluation both require development images.")
    fit, ev = matched[matched.image_id.isin(ids[:n_fit])], matched[matched.image_id.isin(ids[n_fit:])]
    report = []
    for score in cols:
        groups = ["solver", "severity", "size"]
        if score == "measurement_score":
            # The measurement is reused across samples; each paired site counts once.
            fit_scores = fit[fit["sample"] == 0]
            ev_scores = ev[ev["sample"] == 0]
        else:
            fit_scores, ev_scores = fit, ev
        tau = fit_scores.groupby(groups)[f"{score}_off"].quantile(.95).rename("threshold")
        result = ev_scores.merge(tau, on=groups, how="left", validate="many_to_one")
        if result.threshold.isna().any():
            raise ValueError("Calibration misses a solver, severity or size stratum.")
        result["false_alarm"] = result[f"{score}_off"] > result.threshold
        result["detection"] = result[f"{score}_on"] > result.threshold
        result["increment"] = result[f"{score}_on"] - result[f"{score}_off"]
        summary = result.groupby(groups).agg(
            n_images=("image_id", "nunique"), n_score_pairs=("false_alarm", "size"),
            n_false_alarms=("false_alarm", "sum"), n_detections=("detection", "sum"),
            fpr=("false_alarm", "mean"), tpr=("detection", "mean"), median_increment=("increment", "median"),
        ).reset_index()
        report.extend(summary.assign(observer=score).to_dict(orient="records"))
    return {"fit_ids": list(map(int, ids[:n_fit])), "evaluation_ids": list(map(int, ids[n_fit:])), "rows": report}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--runs", nargs="+")
    source.add_argument("--rows", type=Path, help="A previously exported development score table.")
    parser.add_argument("--fit-images", type=int, default=15)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    rows = load_runs(args.runs)[0] if args.runs else pd.read_parquet(args.rows)
    result = assess(rows, args.fit_images)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
