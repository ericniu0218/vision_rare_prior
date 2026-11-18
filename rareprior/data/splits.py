"""Development partitions and legacy holdout definitions."""
from __future__ import annotations

import numpy as np

FFHQ = {
    "validation": list(range(69000, 69100)),   # DPS-vs-official validation only
    "dev": list(range(69100, 69130)),
    "test_pool": list(range(69130, 70000)),
    "unet_train": list(range(60000, 69000)),
}


def ffhq_test(n: int = 100, seed: int = 20261109) -> list[int]:
    rng = np.random.default_rng(seed)
    return sorted(rng.choice(FFHQ["test_pool"], size=n, replace=False).tolist())


def diffusercam_split(n_test_total: int = 999, n_dev: int = 30, n_test: int = 100, seed: int = 7):
    """Indices into the HF DiffuserCam *test* split."""
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n_test_total)
    return {"dev": sorted(perm[:n_dev].tolist()), "test": sorted(perm[n_dev:n_dev + n_test].tolist())}


def knee_subject_split(subjects: list[str], seed: int = 3):
    """14 train / 2 dev / 4 test, by subject (never by slice)."""
    assert len(subjects) == 20, "Stanford 3D FSE knees has 20 volumes"
    rng = np.random.default_rng(seed)
    p = [subjects[i] for i in rng.permutation(20)]
    return {"train": sorted(p[:14]), "dev": sorted(p[14:16]), "test": sorted(p[16:])}


def require_development(split: str, ids=None, kind=None) -> None:
    """Guard experiment entry points while the analysis protocol is draft."""
    if split != "dev":
        raise ValueError("Held-out evaluation is disabled while the protocol is draft; use split=dev.")
    if ids is None:
        return
    allowed = {"ffhq_hf": FFHQ["dev"], "diffusercam_lensed": diffusercam_split()["dev"]}
    if kind in allowed and not set(ids) <= set(allowed[kind]):
        raise ValueError(f"Explicit image IDs must belong to the {kind} development partition.")
    if len(set(ids)) != len(ids):
        raise ValueError("Duplicate image IDs are not allowed.")


def all_disjoint(*groups) -> bool:
    seen = set()
    for g in groups:
        g = set(g)
        if seen & g:
            return False
        seen |= g
    return True
