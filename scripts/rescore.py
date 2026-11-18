"""Recompute site scores with the current observer from saved full reconstructions.

Needed when the observer changes after reconstructions were run (they are the
expensive part). Uses full_<id>.npz (x, xh) and the existing rows for the site
list. Writes a separate derived directory only when every image can be rescored.

  python scripts/rescore.py artifacts/survival/<name>/<hash> artifacts/derived/<name>
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rareprior.features.insert import Site  # noqa: E402
from rareprior.metrics.observer import score_image  # noqa: E402
from rareprior.utils.manifest import atomic_output, git_sha  # noqa: E402


def sites_from_rows(rows: pd.DataFrame) -> list[Site]:
    sites = []
    f = rows[(rows.kind == "feature") & (rows["sample"] == rows["sample"].min())]
    for _, r in f.iterrows():
        sites.append(Site("feature", int(r.cy), int(r.cx), r.cls, int(r["size"]), int(r.variant), int(r.sign),
                          float(r.amplitude), float(r.dprime), float(r.texture),
                          level=float(r.get("level", r.dprime)), dprime_bke=float(r.get("dprime_bke", r.dprime))))
    c = rows[rows.kind == "control"][["cy", "cx", "texture"]].drop_duplicates(["cy", "cx"])
    for _, r in c.iterrows():
        sites.append(Site("control", int(r.cy), int(r.cx), texture=float(r.texture)))
    return sites


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("out", type=Path)
    args = parser.parse_args()
    run = args.run
    manifest = json.loads((run / "manifest.json").read_text())
    if manifest["split"] != "dev" or manifest["config"].get("counterfactual"):
        raise ValueError("Only development runs without counterfactual scores can be rescored. Rerun otherwise.")
    files = sorted(run.glob("rows_*.parquet"))
    if not files or any(not (run / f"full_{f.stem.removeprefix('rows_')}.npz").exists() for f in files):
        raise ValueError("All source images require saved full reconstructions.")
    args.out.mkdir(parents=True, exist_ok=False)
    for f in files:
        img_id = int(f.stem.split("_")[1])
        full = run / f"full_{img_id}.npz"
        rows = pd.read_parquet(f)
        with np.load(full) as z:
            x = torch.from_numpy(z["x"]).double()[None]
            xh = torch.from_numpy(z["xh"].astype(np.float32)).double()
        sites = sites_from_rows(rows)
        keep = ["image_id", "solver", "severity", "dropped"]
        meta = {k: rows[k].iloc[0] for k in keep if k in rows}
        new = []
        for k in range(xh.shape[0]):
            new += score_image(xh[k:k + 1], x, sites, dict(meta, sample=k))
        new = pd.DataFrame(new)
        det_cols = [c for c in rows.columns if c.startswith("det_")]
        if det_cols:
            det = rows[["cy", "cx"] + det_cols].drop_duplicates(["cy", "cx"])
            new = new.merge(det, on=["cy", "cx"], how="left")
        if not rows.kind.isin(["feature", "control"]).all():
            raise ValueError("Source contains scores that require recomputing the full run.")
        shutil.copyfile(run / f"metrics_{img_id}.json", args.out / f"metrics_{img_id}.json")
        with atomic_output(args.out / f.name) as temp:
            new.to_parquet(temp)
    manifest["rescored_at_commit"] = git_sha()
    manifest["rescored_from"] = manifest["config_hash"]
    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Rescored {len(files)} development images into {args.out}")


if __name__ == "__main__":
    main()
