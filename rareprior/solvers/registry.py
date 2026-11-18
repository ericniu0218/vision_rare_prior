"""Build reconstructors from config. Each returns a function
    recon(y, op, x_shape, seeds: list) -> tensor (K, C, H, W)
where seeds has one entry per posterior sample (K = 1 for deterministic solvers)."""
from __future__ import annotations

import torch

from ..diffusion.schedule import Schedule, linear_betas
from .cg import tikhonov
from .diffpir import DiffPIRConfig, diffpir_sample
from .dps import DPSConfig, dps_sample


def _gen(seed, device):
    return torch.Generator(device=device).manual_seed(int(seed))


def build_solver(cfg: dict, device="cpu"):
    name = cfg["name"]
    if name == "tikhonov":
        lam = float(cfg.get("lam", 1e-2))
        iters = int(cfg.get("iters", 100))

        def recon(y, op, x_shape, seeds):
            return tikhonov(op, y, lam, x_shape, iters=iters)
        return recon, 1

    if name == "unet":
        from .unet import load_unet_baseline
        net = load_unet_baseline(cfg["weights"], device)

        def recon(y, op, x_shape, seeds):
            with torch.no_grad():
                return net(op.AT(y, x_shape=x_shape).float()).to(y.real.dtype)
        return recon, 1

    if name in ("dps", "diffpir"):
        from ..diffusion.loader import load_adm
        model = load_adm(cfg.get("prior", "ffhq"), cfg.get("checkpoint"), device=device)
        K = int(cfg.get("K", 4))
        if name == "dps":
            sched = Schedule(linear_betas(1000), respacing=str(cfg.get("steps", 1000)))
            dcfg = DPSConfig(scale=float(cfg.get("scale", 0.3)))

            def recon(y, op, x_shape, seeds):
                outs = []
                y = y.to(torch.complex64 if y.is_complex() else torch.float32)
                for s in seeds:  # batch of one keeps memory low on a T4; seeds give independence
                    g = _gen(s, device)
                    xT = torch.randn((1,) + tuple(x_shape[1:]), generator=g, device=device)
                    outs.append(dps_sample(model, sched, op, y, dcfg, generator=g, x_T=xT))
                return torch.cat(outs)
            return recon, K
        sched = Schedule(linear_betas(1000))
        pcfg = DiffPIRConfig(lam=float(cfg.get("lam", 1.0)), zeta=float(cfg.get("zeta", 0.1)),
                             nfe=int(cfg.get("nfe", 100)), cg_iters=int(cfg.get("cg_iters", 20)))

        def recon(y, op, x_shape, seeds):
            outs = []
            y = y.to(torch.complex64 if y.is_complex() else torch.float32)
            for s in seeds:
                g = _gen(s, device)
                outs.append(diffpir_sample(model, sched, op, y, op.sigma, pcfg, generator=g,
                                           x_shape=(1,) + tuple(x_shape[1:])))
            return torch.cat(outs)
        return recon, K

    raise ValueError(f"unknown solver {name}")
