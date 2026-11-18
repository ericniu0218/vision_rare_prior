"""Survival tables, the H1 contrast and figures from run_survival outputs.

  python scripts/analyze_survival.py --runs artifacts/survival/<name>/<hash> [...] \
      --contrast unet dps --small-only --out results/<name>
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rareprior.features.library import SMALL  # noqa: E402
from rareprior.metrics.observer import add_detection, empirical_fpr, survival_table, thresholds  # noqa: E402
from rareprior.stats.bootstrap import paired_metric_bootstrap, paired_survival_bootstrap  # noqa: E402


def load_runs(dirs):
    rows, mets, arms, design = [], [], set(), {}
    for d in dirs:
        d = Path(d)
        manifest = json.loads((d / "manifest.json").read_text())
        cfg = manifest["config"]
        arms.add((cfg["arm"], manifest["split"], cfg.get("backend")))
        severity = cfg.get("severity", "S1")
        physics = {k: cfg[k] for k in ("operator", "insert", "data") if k in cfg}
        physics.get("operator", {}).pop("blur_backend", None)
        physics["matching_inputs"] = {k: v for k, v in cfg.get("input_sha256", {}).items()
                                      if k in {"power_spectrum", "psf"}}
        if severity in design and design[severity] != physics:
            raise ValueError(f"Runs disagree on the data, operator or insertion design for {severity}.")
        design[severity] = physics
        files = sorted(d.glob("rows_*.parquet"))
        expected = set(cfg.get("image_ids", []))
        seen = {int(f.stem.removeprefix("rows_")) for f in files}
        if not seen or (expected and seen != expected):
            raise ValueError(f"Run is empty or incomplete: {d}")
        for f in files:
            i = int(f.stem.removeprefix("rows_"))
            frame = pd.read_parquet(f)
            metric = json.loads((d / f"metrics_{i}.json").read_text())
            label = cfg["solver"].get("label", cfg["solver"]["name"])
            if (not frame.image_id.eq(i).all() or not frame.solver.eq(label).all()
                    or not frame.severity.eq(severity).all()
                    or metric["image_id"] != i or metric["solver"] != label):
                raise ValueError(f"Result metadata disagrees with the manifest: {f}")
            rows.append(frame)
            mets.append(dict(metric, severity=severity))
    if len(arms) != 1:
        raise ValueError("Runs must share an arm, split and backend.")
    df = pd.concat(rows, ignore_index=True)
    metrics = pd.DataFrame(mets)
    keys = ["image_id", "solver", "severity"]
    if metrics.duplicated(keys).any() or not np.isfinite(df.score).all():
        raise ValueError("Duplicate image results or nonfinite scores.")
    return df, metrics


def check_pairs(df, a, b):
    keys = ["severity", "image_id", "cy", "cx", "cls", "size", "variant", "sign", "level", "amplitude"]
    left, right = (df.loc[(df.solver == s) & (df.kind == "feature"), keys].drop_duplicates() for s in (a, b))
    matched = left.merge(right, on=keys, how="outer", indicator=True)
    if matched.empty or not matched["_merge"].eq("both").all():
        raise ValueError("Contrasted solvers must use the same images, severities and accepted feature sites.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--contrast", nargs=2, metavar=("A", "B"), help="report S_A - S_B")
    ap.add_argument("--small-only", action="store_true")
    ap.add_argument("--n-boot", type=int, default=10000)
    ap.add_argument("--pooling", choices=["site", "macro"], default="site")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    df, mets = load_runs(args.runs)
    main_df = df[df.kind.isin(["feature", "control"])]
    feats = add_detection(main_df)
    if args.small_only:
        feats_s = feats[feats["size"].isin(SMALL)]
    else:
        feats_s = feats
    tab = survival_table(feats_s)
    tab.to_csv(out / "survival_by_cell.csv", index=False)
    summary = {"n_images": int(df.image_id.nunique()),
               "solvers": sorted(df.solver.unique().tolist()),
               "pooled_survival": feats_s.groupby("solver").detected.mean().to_dict(),
               "pooled_ratio_median": feats_s.groupby("solver").ratio.median().to_dict(),
               "dropped_features": int(mets.drop_duplicates("image_id")["dropped"].sum()) if len(mets) else 0}
    if len(mets):
        mets["psnr_mean"] = mets.psnr.map(np.mean)
        mets["ssim_mean"] = mets.ssim.map(np.mean)
        summary["psnr"] = mets.groupby("solver").psnr_mean.mean().to_dict()
        summary["ssim"] = mets.groupby("solver").ssim_mean.mean().to_dict()
        summary["seconds_per_image"] = mets.groupby("solver").seconds.mean().to_dict()
        if "lpips" in mets:
            mets["lpips_mean"] = mets.lpips.dropna().map(np.mean)
            summary["lpips"] = mets.groupby("solver").lpips_mean.mean().to_dict()

    if (df.kind == "cf_null").any():  # control-location null vs counterfactual null (dev check)
        taus = thresholds(main_df)
        cf = df[df.kind == "cf_null"].assign(kind="control")
        summary["counterfactual_fpr"] = empirical_fpr(cf, taus).groupby("solver").fp.mean().to_dict()

    if args.contrast:
        a, b = args.contrast
        check_pairs(main_df, a, b)
        filt = (lambda d: d["size"].isin(SMALL)) if args.small_only else None
        res = paired_survival_bootstrap(main_df, a, b, n_boot=args.n_boot,
                                        cell_filter=filt, pooling=args.pooling)
        summary[f"delta_{a}_minus_{b}"] = res
        if len(mets):
            ma = mets[mets.solver == a].set_index(["image_id", "severity"])
            mb = mets[mets.solver == b].set_index(["image_id", "severity"])
            common = ma.index.intersection(mb.index)
            summary[f"psnr_{a}_minus_{b}"] = paired_metric_bootstrap(
                ma.loc[common].psnr_mean.values, mb.loc[common].psnr_mean.values, n_boot=args.n_boot)

    (out / "summary.json").write_text(json.dumps(summary, indent=2, default=float))
    print(json.dumps(summary, indent=2, default=float))
    try:
        make_plot(tab, out / "survival_curves.png")
    except Exception as e:  # noqa: BLE001
        print("plot skipped:", e)


def make_plot(tab: pd.DataFrame, path: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    solvers = sorted(tab.solver.unique())
    dps = sorted(tab.level.unique())
    fig, axes = plt.subplots(1, len(dps), figsize=(4.2 * len(dps), 3.4), squeeze=False)
    for ax, dp in zip(axes[0], dps):
        t = tab[tab.level == dp]
        g = t.groupby(["solver", "size"]).apply(
            lambda d: np.average(d.survival, weights=d.n), include_groups=False).reset_index(name="s")
        for s in solvers:
            gs = g[g.solver == s]
            ax.plot(gs["size"], gs["s"], marker="o", label=s)
        ax.axhline(np.average(t.ideal_tpr, weights=t.n), ls="--", c="gray", lw=1, label="Gaussian-model reference")
        ax.set_title(f"level = {dp:g}")
        ax.set_xlabel("feature size (px)")
        ax.set_ylim(0, 1.02)
    axes[0][0].set_ylabel("survival (TPR @ 5% FPR)")
    axes[0][-1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)


if __name__ == "__main__":
    main()
