"""Fixed-likelihood toy comparing an exact rare-mode posterior with DPS.

A binary template coefficient is either 0 or 0.8. The likelihood and observed
value stay fixed. We intervene only on prior mass and DPS guidance scale.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import torch
from scipy.special import expit, logit

from rareprior.diffusion.schedule import Schedule, linear_betas
from rareprior.solvers.dps import DPSConfig, dps_sample

PRESENT = 0.8
GAIN = 0.25
SIGMA = 0.10
OBSERVATION = 0.18


def exact_mass(prior):
    log_bayes_factor = (OBSERVATION ** 2 - (OBSERVATION - GAIN * PRESENT) ** 2) / (2 * SIGMA ** 2)
    return float(expit(logit(prior) + log_bayes_factor))


class Measurement:
    def A(self, x):
        return GAIN * x


def denoiser(prior, full):
    acp = torch.from_numpy(full.acp)

    def model(x, t):
        a = acp[t.long()].reshape(-1, 1, 1, 1)
        mu = a.sqrt() * PRESENT
        log_odds = logit(prior) + (2 * x * mu - mu.square()) / (2 * (1 - a))
        mean = PRESENT * torch.sigmoid(log_odds)
        eps = (x - a.sqrt() * mean) / (1 - a).sqrt()
        return torch.cat([eps, torch.zeros_like(eps)], dim=1)

    return model


def trained_denoiser(prior, full, seed, updates):
    """Learn the clean mode from noisy coefficients, then convert to epsilon."""
    torch.manual_seed(seed)
    net = torch.nn.Sequential(torch.nn.Linear(2, 48), torch.nn.SiLU(),
                              torch.nn.Linear(48, 48), torch.nn.SiLU(), torch.nn.Linear(48, 1))
    optimizer = torch.optim.Adam(net.parameters(), lr=0.002)
    g = torch.Generator().manual_seed(seed + 10)
    acp = torch.from_numpy(full.acp).float()

    def logits(x, t):
        a = acp[t]
        log_snr = torch.logit(a)
        return net(torch.stack([x, log_snr / 8], dim=-1)).squeeze(-1)

    for _ in range(updates):
        t = torch.randint(full.T, (256,), generator=g)
        present = (torch.rand(256, generator=g) < prior).float()
        x = acp[t].sqrt() * PRESENT * present + (1 - acp[t]).sqrt() * torch.randn(256, generator=g)
        loss = torch.nn.functional.binary_cross_entropy_with_logits(logits(x, t), present)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
    net.eval()

    def model(x, t):
        probability = logits(x.flatten().float(), t).sigmoid().reshape_as(x)
        a = acp[t].reshape_as(x).double()
        eps = (x - a.sqrt() * PRESENT * probability) / (1 - a).sqrt()
        return torch.cat([eps, torch.zeros_like(eps)], dim=1)

    # Check mode probabilities against the exact score across a fixed forward grid.
    validation = []
    with torch.no_grad():
        for t in (25, 100, 250, 500, 750, 999):
            x = torch.linspace(-2, 2, 401)
            times = torch.full(x.shape, t, dtype=torch.long)
            a = acp[t]
            u = a.sqrt() * PRESENT
            exact = (logit(prior) + (2 * x * u - u.square()) / (2 * (1 - a))).sigmoid()
            validation.append(float((logits(x, times).sigmoid() - exact).abs().mean()))
    return model, max(validation)


def compare(n, steps, learned_updates=0):
    betas = linear_betas(1000)
    full = Schedule(betas)
    schedule = Schedule(betas, respacing=str(steps))
    y = torch.full((n, 1, 1, 1), OBSERVATION, dtype=torch.float64)
    result = []
    for p in (.02, .10, .50):
        seeds = (0, 1, 2) if learned_updates else (None,)
        for training_seed in seeds:
            if learned_updates:
                model, score_error = trained_denoiser(p, full, training_seed, learned_updates)
            else:
                model, score_error = denoiser(p, full), None
            for scale in (0.0, .3, 1.0):
                g = torch.Generator().manual_seed(1)
                # Use the exact terminal mixture with paired uniform and normal draws.
                present = (torch.rand(y.shape, generator=g, dtype=y.dtype) < p).to(y.dtype)
                terminal = float(full.acp[-1])
                xT = math.sqrt(terminal) * PRESENT * present
                xT = xT + math.sqrt(1 - terminal) * torch.randn(y.shape, generator=g, dtype=y.dtype)
                x = dps_sample(model, schedule, Measurement(), y, DPSConfig(scale=scale), generator=g, x_T=xT)
                count = int((x > PRESENT / 2).sum())
                row = dict(prior_mass=p, guidance_scale=scale, exact_posterior_mass=exact_mass(p),
                           reconstructed_present=count, n_chains=n, simulated_dps_mass=count / n)
                if learned_updates:
                    row.update(training_seed=training_seed, max_grid_score_mae=score_error)
                result.append(row)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--chains", type=int, default=4096)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--learned-updates", type=int, default=0,
                        help="Fit three matched scalar denoisers per prior mass instead of using the exact score.")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    summary = dict(kind="synthetic scalar model", observation=OBSERVATION, gain=GAIN, sigma=SIGMA,
                   steps=args.steps, rows=compare(args.chains, args.steps, args.learned_updates))
    if args.learned_updates:
        summary["training_updates"] = args.learned_updates
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(summary, indent=2) + "\n")


if __name__ == "__main__":
    main()
