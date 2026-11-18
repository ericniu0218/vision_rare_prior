"""DPS validation tests that run on CPU.

1. Trajectory equivalence with the official DPS code (tiny random U-Net).
2. Gaussian toy: with an exact score, the sampler reproduces the prior, and DPS
   moves samples toward the analytic posterior mean.
3. Determinism with an explicit generator.
"""
import numpy as np
import pytest
import torch

from rareprior.diffusion.schedule import Schedule, linear_betas, space_timesteps
from rareprior.operators.blur import GaussianBlur
from rareprior.solvers.cg import cg_normal
from rareprior.solvers.dps import DPSConfig, dps_sample

from .conftest import import_official_dps


def test_space_timesteps_matches_official():
    import_official_dps()
    from guided_diffusion.gaussian_diffusion import space_timesteps as official
    for spec in ["20", "100", "50,30", "ddim25", 1000]:
        assert space_timesteps(1000, spec) == sorted(official(1000, spec))


@pytest.mark.parametrize("respacing", ["1000", "20"])
def test_schedule_constants_match_official(respacing):
    import_official_dps()
    from guided_diffusion.gaussian_diffusion import create_sampler
    off = create_sampler("ddpm", 1000, "linear", "epsilon", "learned_range", False, True, False,
                         timestep_respacing=respacing if respacing != "1000" else "")
    ours = Schedule(linear_betas(1000), respacing=respacing)
    assert np.allclose(off.betas, ours.betas, rtol=1e-12)
    assert off.timestep_map == ours.timestep_map


def test_dps_trajectory_matches_official(tiny_unet):
    """Same seed, same model, same measurement: the official loop and ours agree."""
    import_official_dps()
    from guided_diffusion.condition_methods import get_conditioning_method
    from guided_diffusion.gaussian_diffusion import create_sampler

    op = GaussianBlur(kernel_size=7, std=1.0, sigma=0.05)

    class OfficialOp:  # adapter: the official conditioning calls operator.forward
        def forward(self, data, **kw):
            return op.A(data)

    class Noiser:
        __name__ = "gaussian"

    torch.manual_seed(123)
    x_clean = torch.rand(1, 3, 16, 16) * 2 - 1
    y = op.A(x_clean) + 0.05 * torch.randn(1, 3, 16, 16)

    sampler = create_sampler("ddpm", 1000, "linear", "epsilon", "learned_range", False, True, False,
                             timestep_respacing="20")
    cond = get_conditioning_method("ps", OfficialOp(), Noiser(), scale=0.3)
    torch.manual_seed(7)
    x_T = torch.randn(1, 3, 16, 16)
    official = sampler.p_sample_loop(model=tiny_unet, x_start=x_T.clone(), measurement=y,
                                     measurement_cond_fn=cond.conditioning, record=False, save_root=None)

    torch.manual_seed(7)
    x_T2 = torch.randn(1, 3, 16, 16)
    sched = Schedule(linear_betas(1000), respacing="20")
    ours = dps_sample(tiny_unet, sched, op, y, DPSConfig(scale=0.3, official_rng=True), x_T=x_T2)
    err = (official.detach() - ours).abs().max().item()
    assert err < 1e-4, err


class GaussianScore(torch.nn.Module):
    """Exact epsilon-prediction for data x0 ~ N(mu, s^2 I) under the full DDPM schedule."""

    def __init__(self, mu, s, betas):
        super().__init__()
        self.mu, self.s = mu, s
        self.register_buffer("acp", torch.from_numpy(np.cumprod(1 - betas)).float())

    def forward(self, x, t):
        a = self.acp[t].view(-1, 1, 1, 1)
        return torch.sqrt(1 - a) * (x - torch.sqrt(a) * self.mu) / (a * self.s ** 2 + 1 - a)


def test_unconditional_sampler_reproduces_gaussian():
    """With the full 1000-step chain the exact-score sampler reproduces N(0.2, 0.3^2).

    (With fewer respaced steps, fixed-small-variance DDPM under-disperses:
    std 0.251 / 0.272 / 0.288 / 0.296 at 50 / 100 / 250 / 1000 steps. That is
    discretisation, not a bug, and is why DPS step reductions must be validated.)
    """
    betas = linear_betas(1000)
    model = GaussianScore(0.2, 0.3, betas)
    sched = Schedule(betas, respacing="1000", learned_range=False, clip_denoised=False)

    class Zero:  # no measurement: scale 0
        sigma = 0.0

        def A(self, x):
            return x * 0

    g = torch.Generator().manual_seed(0)
    y = torch.zeros(400, 1, 8, 8)
    x = dps_sample(model, sched, Zero(), y, DPSConfig(scale=0.0), generator=g,
                   x_T=torch.randn(400, 1, 8, 8, generator=g))
    assert abs(float(x.mean()) - 0.2) < 0.02
    assert abs(float(x.std()) / 0.3 - 1) < 0.03


def test_respacing_underdisperses():
    betas = linear_betas(1000)
    model = GaussianScore(0.0, 0.3, betas)

    class Zero:
        sigma = 0.0

        def A(self, x):
            return x * 0
    stds = []
    for r in ("50", "1000"):
        g = torch.Generator().manual_seed(0)
        sched = Schedule(betas, respacing=r, learned_range=False, clip_denoised=False)
        x = dps_sample(model, sched, Zero(), torch.zeros(400, 1, 8, 8), DPSConfig(scale=0.0), generator=g,
                       x_T=torch.randn(400, 1, 8, 8, generator=g))
        stds.append(float(x.std()))
    assert stds[0] < 0.9 * stds[1]


def test_dps_moves_toward_analytic_posterior():
    betas = linear_betas(1000)
    mu, s, sigma = 0.0, 0.5, 0.05
    model = GaussianScore(mu, s, betas)
    sched = Schedule(betas, respacing="200", learned_range=False, clip_denoised=False)
    op = GaussianBlur(kernel_size=5, std=1.0, sigma=sigma, channels=1)
    g = torch.Generator().manual_seed(1)
    x0 = mu + s * torch.randn(1, 1, 16, 16, generator=g)
    y = op.A(x0) + sigma * torch.randn(1, 1, 16, 16, generator=g)
    # analytic posterior mean: (A^T A / sig^2 + I / s^2)^-1 (A^T y / sig^2 + mu / s^2)
    rho = sigma ** 2 / s ** 2
    post = cg_normal(op, op.AT(y, x_shape=y.shape) + rho * mu, rho, iters=200)
    K = 32
    ys = y.expand(K, -1, -1, -1).contiguous()
    samples = dps_sample(model, sched, op, ys, DPSConfig(scale=0.5), generator=g)
    prior_samples = dps_sample(model, sched, op, ys, DPSConfig(scale=0.0), generator=g)
    m = samples.mean(0, keepdim=True)
    err_dps = float((m - post).norm())
    err_prior = float((prior_samples.mean(0, keepdim=True) - post).norm())
    corr = float(torch.corrcoef(torch.stack([m.flatten(), post.flatten()]))[0, 1])
    assert err_dps < 0.6 * err_prior
    assert corr > 0.8


def test_dps_deterministic_with_generator(tiny_unet):
    op = GaussianBlur(kernel_size=5, std=1.0, sigma=0.05)
    y = torch.zeros(2, 3, 16, 16)
    sched = Schedule(linear_betas(1000), respacing="10")
    a = dps_sample(tiny_unet, sched, op, y, DPSConfig(), generator=torch.Generator().manual_seed(3))
    b = dps_sample(tiny_unet, sched, op, y, DPSConfig(), generator=torch.Generator().manual_seed(3))
    assert torch.equal(a, b)


def test_per_sample_norm_makes_samples_independent(tiny_unet):
    """Sample 0's trajectory must not depend on what else is in the batch."""
    op = GaussianBlur(kernel_size=5, std=1.0, sigma=0.05)
    sched = Schedule(linear_betas(1000), respacing="10")
    torch.manual_seed(0)
    xT = torch.randn(2, 3, 16, 16)
    y = torch.randn(2, 3, 16, 16) * 0.1

    def run(xT, y):
        return dps_sample(tiny_unet, sched, op, y, DPSConfig(), generator=torch.Generator().manual_seed(5),
                          x_T=xT)
    both = run(xT, y)
    y_alt = y.clone()
    y_alt[1] = 0.5
    both_alt = run(xT, y_alt)
    # the noise draws are batched, so sample 0 sees identical noise in both runs
    assert torch.allclose(both[0], both_alt[0], atol=1e-6)
