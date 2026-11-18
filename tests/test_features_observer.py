import numpy as np
import pandas as pd
import pytest
import torch

from rareprior.features.insert import InsertConfig, amplitude_for_dprime, embed, ideal_tpr, sample_sites
from rareprior.features.library import CLASSES, N_VARIANTS, SIZES, template
from rareprior.metrics.observer import add_detection, score_image, site_score
from rareprior.operators.blur import GaussianBlur
from rareprior.solvers.cg import tikhonov


def test_templates_well_formed():
    for c in CLASSES:
        for s in SIZES[c]:
            for v in range(N_VARIANTS[c]):
                t = template(c, s, v)
                assert t.shape[0] % 2 == 1 and t.shape[0] == t.shape[1]
                assert np.isclose(np.abs(t).max(), 1.0)
                assert (t >= 0).all()
    # glyphs differ from each other
    assert not np.allclose(template("glyph", 9, 0), template("glyph", 9, 1))


def test_insert_hits_target_dprime_and_separation():
    op = GaussianBlur(kernel_size=31, std=3.0, sigma=0.05)
    x = torch.zeros(3, 128, 128)
    cfg = InsertConfig(n_features=4, n_controls=6, margin=16, min_sep=40, ctrl_sep=20)
    x_f, sites, dropped = sample_sites(x, op, np.random.default_rng(0), cfg)
    assert dropped == 0
    feats = [s for s in sites if s.kind == "feature"]
    for s in feats:
        t_full = embed(template(s.cls, s.size, s.variant), s.cy, s.cx, (128, 128), 3)
        d = float(op.A(s.amplitude * t_full).norm()) / op.sigma
        assert d == pytest.approx(s.dprime, rel=1e-4)
    for i, a in enumerate(feats):
        for b in feats[i + 1:]:
            assert (a.cy - b.cy) ** 2 + (a.cx - b.cx) ** 2 >= 40 ** 2
    # same seed -> identical design
    _, sites2, _ = sample_sites(x, op, np.random.default_rng(0), cfg)
    assert [s.to_dict() for s in sites] == [s.to_dict() for s in sites2]


def test_ideal_tpr_matches_monte_carlo():
    rng = np.random.default_rng(0)
    for d in (1.0, 2.0, 4.0):
        null = rng.standard_normal(200000)
        sig = rng.standard_normal(200000) + d
        tau = np.quantile(null, 0.95)
        assert (sig > tau).mean() == pytest.approx(ideal_tpr(d), abs=0.01)


def test_observer_matches_analytic_for_linear_recon():
    """Flat background + Tikhonov: survival has a closed form; the pipeline must match it."""
    H = 48
    op = GaussianBlur(kernel_size=15, std=2.0, sigma=0.05, channels=1)
    lam = 0.05
    t = template("blob", 5)
    cy = cx = H // 2
    t_full = embed(t, cy, cx, (H, H), 1, dtype=torch.float64)
    a = amplitude_for_dprime(op, t_full, 2.0)
    # analytic: s = (a <R A t, t> + <R n, t>) / ||t||, R = (A^T A + lam)^-1 A^T
    RAt = tikhonov(op, op.A(t_full), lam, t_full.shape, iters=300)
    w = embed(t - t.mean(), cy, cx, (H, H), 1, dtype=torch.float64)  # the observer's zero-mean template
    mean = a * float((RAt * w).sum()) / float(w.norm())
    # sd via R^T w = A (A^T A + lam)^-1 w  (R^T = A M^-1 with M symmetric)
    from rareprior.solvers.cg import cg_normal
    Rt_w = op.A(cg_normal(op, w, lam, iters=300))
    sd = op.sigma * float(Rt_w.norm()) / float(w.norm())
    expected = ideal_tpr(mean / sd)

    g = torch.Generator().manual_seed(0)
    rows = []
    for i in range(300):
        for feat in (True, False):
            x = (a * t_full if feat else torch.zeros_like(t_full))
            y = op.A(x) + op.sigma * torch.randn(x.shape, generator=g, dtype=torch.float64)
            xh = tikhonov(op, y, lam, x.shape, iters=60)
            s = site_score(xh[0], t, cy, cx, 1)
            rows.append(dict(kind="feature" if feat else "control", score=s, solver="tik", severity=0,
                             cls="blob", size=5, image_id=i, ratio=0.0, dprime=2.0))
    df = pd.DataFrame(rows)
    feats = add_detection(df)
    assert feats.detected.mean() == pytest.approx(expected, abs=0.07)


def test_score_image_rows():
    op = GaussianBlur(kernel_size=15, std=2.0, sigma=0.05)
    x = torch.zeros(3, 96, 96)
    cfg = InsertConfig(n_features=2, n_controls=2, margin=16, min_sep=30, ctrl_sep=16)
    x_f, sites, _ = sample_sites(x, op, np.random.default_rng(1), cfg)
    rows = score_image(x_f, x, sites, dict(image_id=0, solver="oracle", severity=0))
    feats = [r for r in rows if r["kind"] == "feature"]
    # the oracle reconstruction returns the features exactly: ratio = 1
    for r in feats:
        assert r["ratio"] == pytest.approx(1.0, rel=1e-6)
    n_templates = sum(N_VARIANTS[c] * len(SIZES[c]) for c in CLASSES)
    assert sum(r["kind"] == "control" for r in rows) == 2 * n_templates * 2


def test_border_sites_are_scored():
    diff = torch.zeros(1, 20, 20)
    t = template("blob", 9)  # 25 x 25, larger than the image
    assert site_score(diff, t, 2, 2, 1) == 0.0


def test_bks_insertion_matches_hotelling_dprime(tmp_path):
    from rareprior.features.hotelling import bks_dprime_unit
    op = GaussianBlur(kernel_size=31, std=3.0, sigma=0.05)
    f = torch.fft.fftfreq(128)
    fr2 = (f[:, None] ** 2 + f[None, :] ** 2).clamp_min(1 / 128 ** 2)
    P = (1e-3 / fr2).double()  # 1/f^2 scene spectrum
    path = tmp_path / "P.pt"
    torch.save(P, path)
    cfg = InsertConfig(n_features=3, n_controls=2, margin=16, min_sep=30, ctrl_sep=16, dprimes=(1.0,),
                       dprime_kind="bks", power_spectrum=str(path))
    x = torch.zeros(3, 128, 128)
    _, sites, _ = sample_sites(x, op, np.random.default_rng(0), cfg)
    for s in [s for s in sites if s.kind == "feature"]:
        d_unit = bks_dprime_unit(op, template(s.cls, s.size, s.variant), P)
        assert s.amplitude * d_unit == pytest.approx(1.0, rel=1e-9)
        assert s.dprime_bke > s.dprime  # background variability lowers d'
