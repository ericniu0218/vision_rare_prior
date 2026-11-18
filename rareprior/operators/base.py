"""Linear forward operators: the shared interface used by every solver.

An operator maps an image x (B, C, H, W) to a measurement y. Every operator
provides A (forward), AT (adjoint) and a seeded noise model. The adjoint can be
written in closed form, or obtained exactly by automatic differentiation
(``autograd_adjoint``), which is how we handle reflect-padded convolutions.
"""
from __future__ import annotations

import torch
from torch import nn


class LinearOperator(nn.Module):
    """Base class. Subclasses implement ``A``; ``AT`` defaults to autograd."""

    #: standard deviation of additive Gaussian measurement noise
    sigma: float = 0.0

    def A(self, x: torch.Tensor) -> torch.Tensor:  # noqa: N802 (math notation)
        raise NotImplementedError

    def AT(self, y: torch.Tensor, x_shape=None) -> torch.Tensor:  # noqa: N802
        if x_shape is None:
            raise ValueError("autograd adjoint needs x_shape; override AT for a closed form")
        return autograd_adjoint(self.A, y, x_shape)

    def forward(self, x):
        return self.A(x)

    def add_noise(self, y: torch.Tensor, generator: torch.Generator | None = None) -> torch.Tensor:
        if self.sigma == 0:
            return y
        n = torch.randn(y.shape, generator=generator, dtype=_real_dtype(y), device=y.device)
        if torch.is_complex(y):
            n2 = torch.randn(y.shape, generator=generator, dtype=_real_dtype(y), device=y.device)
            # complex noise with total variance sigma^2 per entry (sigma^2/2 per part)
            return y + self.sigma * (n + 1j * n2) / (2 ** 0.5)
        return y + self.sigma * n

    def normal(self, x: torch.Tensor) -> torch.Tensor:
        """A^T A x."""
        return self.AT(self.A(x), x_shape=x.shape)


def _real_dtype(t: torch.Tensor):
    return t.real.dtype if torch.is_complex(t) else t.dtype


def autograd_adjoint(A, y: torch.Tensor, x_shape) -> torch.Tensor:
    """Exact adjoint of a linear map via a vector-Jacobian product.

    For linear A, d<A(x), y>/dx = A^T y, independent of x.
    """
    with torch.enable_grad():
        x0 = torch.zeros(x_shape, dtype=_real_dtype(y), device=y.device, requires_grad=True)
        out = A(x0)
        inner = (out.conj() * y).real.sum() if torch.is_complex(out) else (out * y).sum()
        (g,) = torch.autograd.grad(inner, x0)
    return g


def inner(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Real inner product <a, b> (real part of a^H b for complex)."""
    if torch.is_complex(a) or torch.is_complex(b):
        return (a.conj() * b).real.sum()
    return (a * b).sum()


def adjoint_error(op: LinearOperator, x_shape, y_like: torch.Tensor | None = None,
                  dtype=torch.float64, seed: int = 0) -> float:
    """Relative error |<Ax, y> - <x, A^T y>| / (||Ax|| ||y||). Should be ~1e-12 in float64."""
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(x_shape, generator=g, dtype=dtype)
    Ax = op.A(x)
    if torch.is_complex(Ax):
        y = torch.complex(torch.randn(Ax.shape, generator=g, dtype=dtype),
                          torch.randn(Ax.shape, generator=g, dtype=dtype))
    else:
        y = torch.randn(Ax.shape, generator=g, dtype=dtype)
    lhs = inner(Ax, y)
    rhs = inner(x, op.AT(y, x_shape=x_shape))
    return float((lhs - rhs).abs() / (Ax.abs().norm() * y.abs().norm()))
