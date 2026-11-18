"""Planted-effect test: fake reconstructors with known survival.

Each fake solver keeps a feature with probability p (else erases it), adds
texture noise at every site, and can hallucinate template-like noise at
control sites. The full pipeline (scores -> per-solver thresholds -> survival
-> clustered bootstrap) must recover what was planted.
"""
import numpy as np
import pandas as pd
import pytest

from rareprior.metrics.observer import add_detection
from rareprior.stats.bootstrap import holm, paired_survival_bootstrap

CELLS = [("blob", 3), ("ring", 5), ("cross", 5)]


def fake_scores(p_keep, n_images=60, n_feat=6, n_ctrl=8, K=2, amp=6.0, noise=1.0, solver="a",
                seed=0, image_offset=0):
    """Scores in units of the matched filter. Kept features score amp + noise,
    erased ones score noise, control sites score noise."""
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n_images):
        img = image_offset + i
        cells = [CELLS[rng.integers(len(CELLS))] for _ in range(n_feat)]
        for k in range(K):
            for (cls, size) in cells:
                kept = rng.random() < p_keep
                rows.append(dict(image_id=img, solver=solver, severity=1, sample=k, kind="feature",
                                 cls=cls, size=size, dprime=2.0, ratio=float(kept),
                                 score=amp * kept + noise * rng.standard_normal()))
            for _ in range(n_ctrl):
                for (cls, size) in CELLS:
                    for sign in (-1, 1):
                        rows.append(dict(image_id=img, solver=solver, severity=1, sample=k, kind="control",
                                         cls=cls, size=size, sign=sign, score=noise * rng.standard_normal()))
    return pd.DataFrame(rows)


def test_recovers_planted_survival():
    # amp large relative to noise: a kept feature is almost always above threshold
    df = fake_scores(0.7, amp=10.0)
    feats = add_detection(df)
    assert feats.detected.mean() == pytest.approx(0.7 + 0.3 * 0.05, abs=0.03)


def test_fpr_is_five_percent():
    df = fake_scores(0.5)
    ctrl = df[df.kind == "control"]
    feats = add_detection(df)
    taus = feats.groupby(["cls", "size"]).tau.first()
    rates = [(ctrl[(ctrl.cls == c) & (ctrl["size"] == s)].score > taus[(c, s)]).mean() for c, s in CELLS]
    assert all(abs(r - 0.05) < 0.01 for r in rates)


@pytest.mark.slow
def test_bootstrap_coverage_planted_difference():
    """95% CIs should cover the true delta in about 95% of replications."""
    truth_a = 0.85 + 0.15 * 0.05
    truth_b = 0.70 + 0.30 * 0.05
    true_delta = truth_a - truth_b
    covered = 0
    reps = 60
    for r in range(reps):
        a = fake_scores(0.85, amp=10, solver="a", seed=1000 + r, n_images=40)
        b = fake_scores(0.70, amp=10, solver="b", seed=5000 + r, n_images=40)
        res = paired_survival_bootstrap(pd.concat([a, b]), "a", "b", n_boot=400, seed=r)
        covered += res["ci_low"] <= true_delta <= res["ci_high"]
    assert 0.85 <= covered / reps <= 1.0


def test_null_difference_is_not_significant():
    a = fake_scores(0.6, solver="a", seed=1)
    b = fake_scores(0.6, solver="b", seed=2)
    res = paired_survival_bootstrap(pd.concat([a, b]), "a", "b", n_boot=1000)
    assert res["ci_low"] < 0 < res["ci_high"]


def test_detects_planted_difference():
    a = fake_scores(0.9, solver="a", seed=1, amp=10)
    b = fake_scores(0.6, solver="b", seed=2, amp=10)
    res = paired_survival_bootstrap(pd.concat([a, b]), "a", "b", n_boot=1000)
    assert res["ci_low"] > 0.15
    assert res["p_value"] < 0.01


def test_hallucination_is_charged_by_per_solver_threshold():
    """A solver with noisier controls (hallucination) gets a higher threshold and lower survival."""
    clean = fake_scores(0.8, amp=4.0, noise=1.0, solver="clean", seed=3)
    noisy = fake_scores(0.8, amp=4.0, noise=2.0, solver="noisy", seed=3)
    f = add_detection(pd.concat([clean, noisy]), group=["solver", "severity", "cls", "size"])
    s = f.groupby("solver").detected.mean()
    assert s["clean"] > s["noisy"] + 0.1


def test_holm():
    out = holm({"h2": 0.01, "h3a": 0.04, "h3b": 0.03, "h4": 0.2})
    assert out["h2"][1] is True       # 0.01 <= 0.05/4
    assert out["h3b"][1] is False     # 0.03 > 0.05/3 -> stop
    assert out["h3a"][1] is False and out["h4"][1] is False
