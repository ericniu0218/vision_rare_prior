import json

import pytest
import torch

import numpy as np

from rareprior.data.splits import FFHQ, all_disjoint, diffusercam_split, ffhq_test
from rareprior.diffusion.schedule import Schedule, linear_betas
from rareprior.operators.blur import GaussianBlur
from rareprior.optics.psf import (FreeFormPSF, PhaseMaskPSF, optimize_psf, participation_ratio,
                                  scene_power_spectrum, summarize, template_spectra)
from rareprior.solvers.diffpir import DiffPIRConfig, diffpir_sample, diffpir_timesteps
from rareprior.utils.manifest import config_hash, write_manifest


def test_splits_disjoint():
    test = ffhq_test()
    assert len(test) == 100
    assert all_disjoint(FFHQ["validation"], FFHQ["dev"], test, FFHQ["unet_train"])
    assert set(test) <= set(FFHQ["test_pool"])
    dc = diffusercam_split()
    assert all_disjoint(dc["dev"], dc["test"])
    assert ffhq_test() == test  # frozen seed


def test_manifest_and_audit(tmp_path):
    cfg = {"a": 1, "b": [1, 2]}
    assert config_hash(cfg) == config_hash({"b": [1, 2], "a": 1})
    assert config_hash(cfg) != config_hash({"a": 2, "b": [1, 2]})
    log = tmp_path / "test_log.jsonl"
    m = write_manifest(tmp_path, cfg, split="test", audit_log=log)
    assert json.loads((tmp_path / "manifest.json").read_text())["config_hash"] == m["config_hash"]
    assert len(log.read_text().splitlines()) == 1
    write_manifest(tmp_path, cfg, split="dev", audit_log=log)
    assert len(log.read_text().splitlines()) == 1  # dev runs are not logged


def test_psf_constraints():
    h = FreeFormPSF((16, 16))()
    assert float(h.min()) >= 0 and float(h.sum()) == pytest.approx(1.0)
    pm = PhaseMaskPSF((32, 32), coarse=8)
    h2 = pm()
    assert float(h2.min()) >= 0 and float(h2.sum()) == pytest.approx(1.0)
    # angular spectrum propagation conserves energy for propagating waves
    u0 = torch.exp(1j * pm.phase())
    uz = torch.fft.ifft2(torch.fft.fft2(u0) * pm.transfer)
    assert float((uz.abs() ** 2).sum()) <= float((u0.abs() ** 2).sum()) * (1 + 1e-9)


def test_psf_objectives_behave():
    g = torch.Generator().manual_seed(0)
    scenes = torch.randn(8, 16, 16, generator=g, dtype=torch.float64).cumsum(-1).cumsum(-2) * 0.05
    pad = (32, 32)
    P = scene_power_spectrum(scenes, pad)
    T2 = template_spectra(pad)
    delta = torch.zeros(16, 16, dtype=torch.float64)
    delta[8, 8] = 1
    flat = torch.full((16, 16), 1 / 256, dtype=torch.float64)
    s_delta = summarize(delta, pad, P, 1e-4, T2)
    s_flat = summarize(flat, pad, P, 1e-4, T2)
    # a delta (perfect lens) beats a flat smear on every objective
    assert s_delta["mse"] < s_flat["mse"]
    assert s_delta["mi_nats"] > s_flat["mi_nats"]
    assert s_delta["detect_d2"] > s_flat["detect_d2"]
    assert participation_ratio(delta) < participation_ratio(flat)
    # optimisation improves its own objective
    param = FreeFormPSF((16, 16), init=flat + 1e-3 * torch.rand(16, 16, generator=g, dtype=torch.float64))
    h, hist = optimize_psf(param, "detect", pad, P, 1e-4, T2, steps=150, lr=0.1, log_every=50)
    assert hist[-1][1] < hist[0][1]


def test_diffpir_timesteps_descending():
    ts = diffpir_timesteps(1000, 100)
    assert ts[0] == 999 and ts[-1] >= 0 and all(a > b for a, b in zip(ts, ts[1:]))
    assert len(ts) == 100


def test_diffpir_runs_and_fits_data(tiny_unet):
    op = GaussianBlur(kernel_size=5, std=1.0, sigma=0.05)
    g = torch.Generator().manual_seed(0)
    x = torch.rand(1, 3, 16, 16, generator=g) * 2 - 1
    y = op.A(x) + 0.05 * torch.randn(1, 3, 16, 16, generator=g)
    sched = Schedule(linear_betas(1000))
    xh = diffpir_sample(tiny_unet, sched, op, y, 0.05, DiffPIRConfig(nfe=10, lam=50.0), generator=g)
    assert torch.isfinite(xh).all()
    assert float((op.A(xh) - y).norm()) < float((op.A(torch.zeros_like(x)) - y).norm())


def test_surprisal_zero_for_identical_images(tiny_unet):
    from rareprior.features.surprisal import minority_score, surprisal
    sched = Schedule(linear_betas(1000))
    x = torch.zeros(1, 3, 16, 16)
    g = torch.Generator().manual_seed(0)
    assert abs(surprisal(tiny_unet, sched, x, x, 8, 8, r=4, timesteps=(100, 500), n_noise=2,
                         generator=g)) < 1e-9
    assert abs(minority_score(tiny_unet, sched, x, x, 8, 8, r=4, n_noise=2, generator=g)) < 1e-9


def test_auroc_matches_definition():
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
    from analyze_detector import auroc
    s = np.array([0.1, 0.4, 0.35, 0.8])
    y = np.array([0, 0, 1, 1])
    assert auroc(s, y) == 0.75
    assert auroc(np.array([1.0, 1.0]), np.array([0, 1])) == 0.5
