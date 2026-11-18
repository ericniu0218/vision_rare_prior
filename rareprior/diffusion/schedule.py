"""DDPM noise schedule, timestep respacing and the reverse-step mean/variance.

Written from the equations in Ho et al. (2020) and Nichol & Dhariwal (2021),
and checked to match the official DPS code (guided_diffusion/gaussian_diffusion.py
and posterior_mean_variance.py, commit effbde7) to float32 precision:
  * constants are computed in float64 numpy and cast to float32 when indexed;
  * epsilon prediction, pred_xstart clipped to [-1, 1];
  * learned_range variance: interpolate log-variance between the clipped
    posterior variance and beta_t using the model's second output half.
"""
from __future__ import annotations

import math

import numpy as np
import torch


def linear_betas(T: int = 1000) -> np.ndarray:
    scale = 1000 / T
    return np.linspace(scale * 1e-4, scale * 0.02, T, dtype=np.float64)


def cosine_betas(T: int = 1000, max_beta: float = 0.999) -> np.ndarray:
    f = lambda t: math.cos((t + 0.008) / 1.008 * math.pi / 2) ** 2  # noqa: E731
    return np.array([min(1 - f((i + 1) / T) / f(i / T), max_beta) for i in range(T)])


def space_timesteps(num_timesteps: int, section_counts) -> list[int]:
    """Same semantics as guided_diffusion.space_timesteps; returns a sorted list."""
    if isinstance(section_counts, str):
        if section_counts.startswith("ddim"):
            want = int(section_counts[4:])
            for i in range(1, num_timesteps):
                if len(range(0, num_timesteps, i)) == want:
                    return list(range(0, num_timesteps, i))
            raise ValueError("cannot create exact ddim stride")
        section_counts = [int(x) for x in section_counts.split(",")]
    elif isinstance(section_counts, int):
        section_counts = [section_counts]
    size_per, extra = divmod(num_timesteps, len(section_counts))
    start, steps = 0, []
    for i, count in enumerate(section_counts):
        size = size_per + (1 if i < extra else 0)
        if size < count:
            raise ValueError(f"cannot divide section of {size} steps into {count}")
        stride = 1 if count <= 1 else (size - 1) / (count - 1)
        cur = 0.0
        for _ in range(count):
            steps.append(start + round(cur))
            cur += stride
        start += size
    return sorted(set(steps))


class Schedule:
    """Holds (possibly respaced) DDPM constants. ``timestep_map[i]`` is the
    original timestep fed to the network at respaced index i."""

    def __init__(self, betas: np.ndarray, respacing=None, clip_denoised: bool = True,
                 learned_range: bool = True):
        betas = np.asarray(betas, dtype=np.float64)
        T0 = len(betas)
        use = list(range(T0)) if respacing in (None, T0, str(T0)) else space_timesteps(T0, respacing)
        acp_full = np.cumprod(1.0 - betas)
        last, new_betas = 1.0, []
        for i in use:
            new_betas.append(1 - acp_full[i] / last)
            last = acp_full[i]
        b = np.array(new_betas, dtype=np.float64)
        self.timestep_map = use
        self.betas = b
        self.T = len(b)
        self.clip_denoised = clip_denoised
        self.learned_range = learned_range
        a = 1.0 - b
        self.acp = np.cumprod(a)
        self.acp_prev = np.append(1.0, self.acp[:-1])
        self.sqrt_recip_acp = np.sqrt(1.0 / self.acp)
        self.sqrt_recipm1_acp = np.sqrt(1.0 / self.acp - 1)
        self.post_var = b * (1.0 - self.acp_prev) / (1.0 - self.acp)
        self.post_logvar_clipped = np.log(np.append(self.post_var[1], self.post_var[1:]))
        self.coef1 = b * np.sqrt(self.acp_prev) / (1.0 - self.acp)
        self.coef2 = (1.0 - self.acp_prev) * np.sqrt(a) / (1.0 - self.acp)

    # -- helpers -----------------------------------------------------------
    @staticmethod
    def _ex(arr: np.ndarray, t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        v = torch.from_numpy(arr).to(device=x.device, dtype=torch.float32)[t]
        while v.ndim < x.ndim:
            v = v.unsqueeze(-1)
        return v.expand_as(x).to(x.dtype)

    def model_t(self, t: torch.Tensor) -> torch.Tensor:
        return torch.tensor(self.timestep_map, device=t.device, dtype=t.dtype)[t]

    def eps_and_var(self, model, x, t):
        out = model(x, self.model_t(t))
        C = x.shape[1]
        if out.shape[1] == 2 * C:
            eps, v = torch.split(out, C, dim=1)
        else:
            eps, v = out, None
        return eps, v

    def pred_xstart(self, x, t, eps):
        x0 = self._ex(self.sqrt_recip_acp, t, x) * x - self._ex(self.sqrt_recipm1_acp, t, eps) * eps
        return x0.clamp(-1, 1) if self.clip_denoised else x0

    def p_mean_variance(self, model, x, t):
        eps, v = self.eps_and_var(model, x, t)
        x0 = self.pred_xstart(x, t, eps)
        mean = self._ex(self.coef1, t, x) * x0 + self._ex(self.coef2, t, x) * x
        if v is not None and self.learned_range:
            min_log = self._ex(self.post_logvar_clipped, t, x)
            max_log = self._ex(np.log(self.betas), t, x)
            frac = (v + 1.0) / 2.0
            logvar = frac * max_log + (1 - frac) * min_log
        else:  # fixed_small
            logvar = self._ex(np.log(np.append(self.post_var[1], self.post_var[1:])), t, x)
        return {"mean": mean, "log_variance": logvar, "pred_xstart": x0, "eps": eps}

    def q_sample(self, x0, t, noise):
        return self._ex(np.sqrt(self.acp), t, x0) * x0 + self._ex(np.sqrt(1 - self.acp), t, x0) * noise
