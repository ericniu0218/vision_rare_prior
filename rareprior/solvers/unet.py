"""MSE-trained U-Net baseline (non-generative). Input: A^T y. Output: x_hat.

The network learns a residual on top of its input. It is trained on clean
images that never contain inserted features, with the same operator and
noise level as at test time (see scripts/train_unet.py).
"""
from __future__ import annotations

import torch
from torch import nn


def _block(cin, cout):
    return nn.Sequential(
        nn.Conv2d(cin, cout, 3, padding=1), nn.GroupNorm(8, cout), nn.SiLU(),
        nn.Conv2d(cout, cout, 3, padding=1), nn.GroupNorm(8, cout), nn.SiLU())


class UNet(nn.Module):
    def __init__(self, channels: int = 3, base: int = 48, levels: int = 4):
        super().__init__()
        chs = [base * 2 ** i for i in range(levels)]
        self.inc = _block(channels, chs[0])
        self.downs = nn.ModuleList([_block(chs[i], chs[i + 1]) for i in range(levels - 1)])
        self.pool = nn.AvgPool2d(2)
        self.ups = nn.ModuleList([nn.ConvTranspose2d(chs[i + 1], chs[i], 2, stride=2)
                                  for i in reversed(range(levels - 1))])
        self.up_blocks = nn.ModuleList([_block(2 * chs[i], chs[i]) for i in reversed(range(levels - 1))])
        self.out = nn.Conv2d(chs[0], channels, 1)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)
        self.config = dict(channels=channels, base=base, levels=levels)

    def forward(self, x):
        skips = [self.inc(x)]
        h = skips[0]
        for d in self.downs:
            h = d(self.pool(h))
            skips.append(h)
        h = skips.pop()
        for up, blk in zip(self.ups, self.up_blocks):
            h = blk(torch.cat([up(h), skips.pop()], dim=1))
        return x + self.out(h)


def load_unet_baseline(path, device="cpu") -> UNet:
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    net = UNet(**ckpt["config"])
    net.load_state_dict(ckpt["state_dict"])
    return net.to(device).eval()


def n_params(m: nn.Module) -> int:
    return sum(p.numel() for p in m.parameters())
