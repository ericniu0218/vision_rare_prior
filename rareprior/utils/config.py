"""Tiny config system: YAML files plus dotted command-line overrides.

    python scripts/run_survival.py configs/experiment/ffhq_blur_dev.yaml solver.name=tikhonov n_images=5

(The plan originally named Hydra; this keeps dependencies minimal and the
behaviour explicit. Every resolved config is saved in the stage manifest.)
"""
from __future__ import annotations

import copy
from pathlib import Path

import yaml


def _set(d: dict, dotted: str, value):
    keys = dotted.split(".")
    for k in keys[:-1]:
        d = d.setdefault(k, {})
    d[keys[-1]] = value


def load_config(path: str | Path, overrides: list[str] = ()) -> dict:
    cfg = yaml.safe_load(Path(path).read_text()) or {}
    base = cfg.pop("defaults", None)
    if base:
        parent = load_config(Path(path).parent / base)
        cfg = deep_merge(parent, cfg)
    for o in overrides:
        k, v = o.split("=", 1)
        _set(cfg, k, yaml.safe_load(v))
    return cfg


def deep_merge(a: dict, b: dict) -> dict:
    out = copy.deepcopy(a)
    for k, v in b.items():
        out[k] = deep_merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else v
    return out
