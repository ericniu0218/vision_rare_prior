"""Conjugate gradient for (A^T A + rho I) x = b, batched over the first dimension."""
from __future__ import annotations

import torch


def _bdot(a, b):
    return (a * b).reshape(a.shape[0], -1).sum(1).view(-1, *([1] * (a.ndim - 1)))


def cg_normal(op, b, rho, x0=None, iters: int = 20, tol: float = 1e-6):
    """Solve (A^T A + rho I) x = b. rho: float or tensor broadcastable to b."""
    def M(v):
        return op.AT(op.A(v), x_shape=v.shape) + rho * v

    x = torch.zeros_like(b) if x0 is None else x0.clone()
    r = b - M(x)
    p = r.clone()
    rs = _bdot(r, r)
    b_norm = _bdot(b, b).sqrt().clamp_min(1e-30)
    for _ in range(iters):
        Mp = M(p)
        alpha = rs / _bdot(p, Mp).clamp_min(1e-30)
        x = x + alpha * p
        r = r - alpha * Mp
        rs_new = _bdot(r, r)
        if bool((rs_new.sqrt() / b_norm).max() < tol):
            break
        p = r + (rs_new / rs.clamp_min(1e-30)) * p
        rs = rs_new
    return x


def tikhonov(op, y, lam: float, x_shape, iters: int = 100, x0=None):
    """Linear (Wiener-like) reconstruction: argmin ||y - Ax||^2 + lam ||x||^2."""
    b = op.AT(y, x_shape=x_shape)
    return cg_normal(op, b, lam, x0=x0, iters=iters)
