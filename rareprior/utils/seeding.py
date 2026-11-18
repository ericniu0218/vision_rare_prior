"""Deterministic per-item seeds so every solver sees identical features and noise."""
from __future__ import annotations

import hashlib

import numpy as np
import torch


def item_seed(*parts) -> int:
    h = hashlib.sha256("/".join(str(p) for p in parts).encode()).digest()
    return int.from_bytes(h[:4], "little")


def rng(*parts) -> np.random.Generator:
    return np.random.default_rng(item_seed(*parts))


def torch_gen(*parts, device="cpu") -> torch.Generator:
    return torch.Generator(device=device).manual_seed(item_seed(*parts))
