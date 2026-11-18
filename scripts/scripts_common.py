"""Shared helpers for analysis scripts."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


def load_rows(dirs) -> pd.DataFrame:
    frames = []
    for d in dirs:
        for f in sorted(Path(d).glob("rows_*.parquet")):
            frames.append(pd.read_parquet(f))
    if not frames:
        raise FileNotFoundError(f"no rows_*.parquet in {dirs}")
    return pd.concat(frames, ignore_index=True)


def load_metrics(dirs) -> pd.DataFrame:
    rows = []
    for d in dirs:
        for f in sorted(Path(d).glob("metrics_*.json")):
            rows.append(json.loads(f.read_text()))
    return pd.DataFrame(rows)
