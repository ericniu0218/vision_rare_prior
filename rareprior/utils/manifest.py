"""Stage manifests, config hashing and the test-split audit log."""
from __future__ import annotations

import hashlib
import json
import platform
import subprocess
import tempfile
from contextlib import contextmanager
import time
from pathlib import Path


def config_hash(cfg: dict) -> str:
    blob = json.dumps(cfg, sort_keys=True, default=str).encode()
    return hashlib.sha256(blob).hexdigest()[:12]


def git_sha() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL,
                                       text=True).strip()
    except Exception:  # noqa: BLE001
        return "unknown"


def library_versions() -> dict:
    out = {"python": platform.python_version()}
    for name in ("torch", "numpy", "scipy", "sigpy", "pandas", "statsmodels"):
        try:
            out[name] = __import__(name).__version__
        except Exception:  # noqa: BLE001
            out[name] = None
    return out


def stage_dir(root: str | Path, stage: str, cfg: dict) -> Path:
    d = Path(root) / stage / config_hash(cfg)
    d.mkdir(parents=True, exist_ok=True)
    return d


@contextmanager
def atomic_output(path: Path):
    """Publish a completed file with a same-directory atomic rename."""
    with tempfile.NamedTemporaryFile(dir=path.parent, suffix=path.suffix, delete=False) as handle:
        temp = Path(handle.name)
    try:
        yield temp
        temp.replace(path)
    finally:
        temp.unlink(missing_ok=True)


def write_manifest(out_dir: Path, cfg: dict, split: str, extra: dict | None = None,
                   audit_log: str | Path | None = None) -> dict:
    import torch
    m = {
        "config": cfg,
        "config_hash": config_hash(cfg),
        "split": split,
        "git_sha": git_sha(),
        "versions": library_versions(),
        "device": (torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"),
        "time_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    if extra:
        m.update(extra)
    with atomic_output(Path(out_dir) / "manifest.json") as temp:
        temp.write_text(json.dumps(m, indent=2, default=str))
    if split == "test" and audit_log is not None:
        with open(audit_log, "a") as f:
            f.write(json.dumps({k: m[k] for k in ("config_hash", "git_sha", "time_utc")} |
                               {"stage": str(out_dir)}) + "\n")
    return m
