"""PSNR, SSIM and (optional) LPIPS for images in [-1, 1]."""
from __future__ import annotations

import torch
import torch.nn.functional as F


def psnr(x: torch.Tensor, ref: torch.Tensor, data_range: float = 2.0) -> torch.Tensor:
    mse = ((x - ref) ** 2).reshape(x.shape[0], -1).mean(1)
    return 10 * torch.log10(data_range ** 2 / mse.clamp_min(1e-20))


def _gauss(size=11, sigma=1.5, dtype=torch.float64):
    g = torch.arange(size, dtype=dtype) - size // 2
    g = torch.exp(-g ** 2 / (2 * sigma ** 2))
    g = g / g.sum()
    return g[:, None] * g[None, :]


def ssim(x: torch.Tensor, ref: torch.Tensor, data_range: float = 2.0) -> torch.Tensor:
    """Standard Gaussian-window SSIM (Wang et al. 2004), mean over channels."""
    C = x.shape[1]
    w = _gauss(dtype=x.dtype).to(x.device).expand(C, 1, -1, -1)
    c1, c2 = (0.01 * data_range) ** 2, (0.03 * data_range) ** 2
    mu_x = F.conv2d(x, w, groups=C)
    mu_y = F.conv2d(ref, w, groups=C)
    sxx = F.conv2d(x * x, w, groups=C) - mu_x ** 2
    syy = F.conv2d(ref * ref, w, groups=C) - mu_y ** 2
    sxy = F.conv2d(x * ref, w, groups=C) - mu_x * mu_y
    m = ((2 * mu_x * mu_y + c1) * (2 * sxy + c2)) / ((mu_x ** 2 + mu_y ** 2 + c1) * (sxx + syy + c2))
    return m.reshape(x.shape[0], -1).mean(1)


_LPIPS = None


def lpips_distance(x: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
    """LPIPS (AlexNet). Requires the `lpips` package and its weights."""
    global _LPIPS
    if _LPIPS is None:
        import lpips
        _LPIPS = lpips.LPIPS(net="alex", verbose=False).eval()
    _LPIPS.to(x.device)
    with torch.no_grad():
        return _LPIPS(x.float(), ref.float()).flatten()
