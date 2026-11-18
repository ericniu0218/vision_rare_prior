"""How unusual is a feature to the diffusion prior? (H2)

surprisal: increase in the prior's denoising loss inside a window around the
    feature, caused by adding the feature, averaged over timesteps and noise
    draws (the same noise for both images, so the difference has low variance).
minority_score: the Um, Lee & Ye (ICLR 2024) idea applied to a window: the
    distance between an image and its one-step Tweedie reconstruction from a
    noised version, with minus without the feature. They use LPIPS on whole
    images; here an L2 version on the window is the default (LPIPS optional).
"""
from __future__ import annotations

import torch

from ..diffusion.schedule import Schedule


def _window_mask(shape, cy, cx, r, device):
    m = torch.zeros(shape[-2:], device=device)
    m[max(0, cy - r):cy + r + 1, max(0, cx - r):cx + r + 1] = 1
    return m


@torch.no_grad()
def surprisal(model, sched: Schedule, x: torch.Tensor, x_feat: torch.Tensor, cy: int, cx: int, r: int = 12,
              timesteps=(100, 200, 300, 400, 500, 600, 700, 800, 900), n_noise: int = 4,
              generator: torch.Generator | None = None) -> float:
    """x, x_feat: (1, C, H, W) in [-1, 1]. Timesteps are original indices (full schedule)."""
    assert sched.timestep_map == list(range(sched.T)), "use the full 1000-step schedule"
    m = _window_mask(x.shape, cy, cx, r, x.device)
    tot = 0.0
    for t in timesteps:
        tt = torch.full((1,), t, device=x.device, dtype=torch.long)
        for _ in range(n_noise):
            eps = torch.randn(x.shape, generator=generator, device=x.device, dtype=x.dtype)
            l1 = ((sched.eps_and_var(model, sched.q_sample(x_feat, tt, eps), tt)[0] - eps) ** 2 * m).sum()
            l0 = ((sched.eps_and_var(model, sched.q_sample(x, tt, eps), tt)[0] - eps) ** 2 * m).sum()
            tot += float(l1 - l0)
    return tot / (len(timesteps) * n_noise)


@torch.no_grad()
def minority_score(model, sched: Schedule, x: torch.Tensor, x_feat: torch.Tensor, cy: int, cx: int,
                   r: int = 12, t: int = 500, n_noise: int = 4, generator=None) -> float:
    m = _window_mask(x.shape, cy, cx, r, x.device)
    tt = torch.full((1,), t, device=x.device, dtype=torch.long)
    tot = 0.0
    for _ in range(n_noise):
        eps = torch.randn(x.shape, generator=generator, device=x.device, dtype=x.dtype)
        out = []
        for img in (x_feat, x):
            xt = sched.q_sample(img, tt, eps)
            x0 = sched.pred_xstart(xt, tt, sched.eps_and_var(model, xt, tt)[0])
            out.append(float((((x0 - img) ** 2) * m).sum()))
        tot += out[0] - out[1]
    return tot / n_noise
