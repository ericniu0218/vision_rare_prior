"""Rare-feature templates: blob, ring, cross, glyph, line.

Each template is a small 2D float64 array with peak |value| = 1, rendered with
8x supersampling and box downsampling so that thin shapes are anti-aliased
and identical across platforms. Glyphs come from a 5x7 bitmap font defined
here (no system fonts), so results don't depend on what is installed.
"""
from __future__ import annotations

import math
from functools import lru_cache

import numpy as np

CLASSES = ("blob", "ring", "cross", "glyph", "line")
SIZES = {"blob": (3, 5, 9), "ring": (3, 5, 9), "cross": (3, 5, 9), "glyph": (5, 9), "line": (3, 5, 9)}
SMALL = (3, 5)
N_VARIANTS = {"blob": 1, "ring": 1, "cross": 1, "glyph": 5, "line": 8}
SS = 8  # supersampling factor

_FONT = {  # 5 columns x 7 rows, '#' = on
    "A": [".###.", "#...#", "#...#", "#####", "#...#", "#...#", "#...#"],
    "E": ["#####", "#....", "#....", "####.", "#....", "#....", "#####"],
    "K": ["#...#", "#..#.", "#.#..", "##...", "#.#..", "#..#.", "#...#"],
    "R": ["####.", "#...#", "#...#", "####.", "#.#..", "#..#.", "#...#"],
    "X": ["#...#", "#...#", ".#.#.", "..#..", ".#.#.", "#...#", "#...#"],
}
GLYPHS = tuple(_FONT)


def _downsample(a: np.ndarray, f: int) -> np.ndarray:
    h, w = a.shape[0] // f, a.shape[1] // f
    return a[:h * f, :w * f].reshape(h, f, w, f).mean(axis=(1, 3))


def _grid(n: int):
    """Supersampled pixel-centre coordinates of an n x n patch, centred at 0."""
    c = (np.arange(n * SS) + 0.5) / SS - n / 2
    return np.meshgrid(c, c, indexing="ij")


def _normalize(a: np.ndarray) -> np.ndarray:
    m = np.abs(a).max()
    return a / m if m > 0 else a


@lru_cache(maxsize=None)
def template(cls: str, size: int, variant: int = 0) -> np.ndarray:
    """Return the template array (odd side length, centred)."""
    if cls not in CLASSES or size not in SIZES[cls]:
        raise ValueError(f"unknown template {cls}/{size}")
    if cls == "blob":
        std = size / 2.355  # FWHM = size
        n = 2 * math.ceil(3 * std) + 1
        yy, xx = _grid(n)
        a = np.exp(-(yy ** 2 + xx ** 2) / (2 * std ** 2))
    elif cls == "ring":
        n = size + 2 + (size % 2 == 0)
        yy, xx = _grid(n)
        r = np.sqrt(yy ** 2 + xx ** 2)
        a = (np.abs(r - (size - 1) / 2) <= 0.5).astype(float)
    elif cls == "cross":
        n = size + (size % 2 == 0)
        yy, xx = _grid(n)
        half = size / 2
        a = (((np.abs(yy) <= 0.5) & (np.abs(xx) <= half)) |
             ((np.abs(xx) <= 0.5) & (np.abs(yy) <= half))).astype(float)
    elif cls == "glyph":
        bitmap = np.array([[c == "#" for c in row] for row in _FONT[GLYPHS[variant % 5]]], float)
        scale = size / 5.0
        h = int(round(7 * scale))
        w = int(round(5 * scale))
        n = max(h, w) + 2
        n += (n % 2 == 0)
        yy, xx = _grid(n)
        # map supersampled coords to bitmap cells (glyph centred in the patch)
        col = np.floor((xx + w / 2) / scale).astype(int)
        row = np.floor((yy + h / 2) / scale).astype(int)
        ok = (row >= 0) & (row < 7) & (col >= 0) & (col < 5)
        a = np.zeros_like(xx)
        a[ok] = bitmap[row[ok], col[ok]]
    else:  # line, 8 orientations
        theta = math.pi * (variant % 8) / 8
        n = size + 2 + (size % 2 == 0)
        yy, xx = _grid(n)
        along = xx * math.cos(theta) + yy * math.sin(theta)
        across = -xx * math.sin(theta) + yy * math.cos(theta)
        a = ((np.abs(across) <= 0.5) & (np.abs(along) <= size / 2)).astype(float)
    out = _normalize(_downsample(a, SS))
    assert out.shape[0] % 2 == 1 and out.shape[0] == out.shape[1]
    return out


def all_cells(sizes_filter=None):
    """List of (cls, size) cells in the design."""
    cells = []
    for c in CLASSES:
        for s in SIZES[c]:
            if sizes_filter is None or s in sizes_filter:
                cells.append((c, s))
    return cells
