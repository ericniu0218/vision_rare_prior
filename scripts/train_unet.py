"""Train the MSE U-Net baseline on simulated measurements (no inserted features).

  python scripts/train_unet.py configs/experiment/<name>.yaml unet.steps=20000
Images come from the same `data` block as run_survival (training indices are
taken from the split's training pool). Saves <out>/unet.pt and a manifest.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rareprior.solvers.unet import UNet, n_params  # noqa: E402
from rareprior.utils.config import load_config  # noqa: E402
from rareprior.utils.manifest import atomic_output, git_sha, stage_dir, write_manifest  # noqa: E402

from run_survival import build_operator, synthetic_image  # noqa: E402


def training_images(cfg):
    d = cfg["data"]
    n = cfg["unet"].get("n_train", 9000)
    if d["kind"] == "synthetic":
        return [synthetic_image(10_000 + i, d.get("size", 256), d.get("channels", 3)) for i in range(n)]
    if d["kind"] == "ffhq_hf":
        from rareprior.data import splits
        from rareprior.data.loaders import load_ffhq
        ids = splits.FFHQ["unet_train"][:n]
        images = load_ffhq(ids, repo=d.get("repo", "merkol/ffhq-256"), size=d.get("size", 256),
                           revision=d.get("revision"))
        return [images[i] for i in ids]
    if d["kind"] == "diffusercam_lensed":
        raise NotImplementedError("use the HF train split; see the plan, Section 8.2")
    raise ValueError(d["kind"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("config")
    ap.add_argument("overrides", nargs="*")
    args = ap.parse_args()
    cfg = load_config(args.config, args.overrides)
    u = cfg["unet"]
    device = cfg.get("device", "cuda" if torch.cuda.is_available() else "cpu")
    seed = int(u.get("seed", 0))
    run_cfg = {"data": cfg["data"], "operator": cfg["operator"], "unet": dict(u, seed=seed),
               "code_commit": git_sha(), "backend": torch.device(device).type, "torch_version": str(torch.__version__)}
    out = stage_dir(cfg.get("out_root", "artifacts"), f"unet/{cfg['name']}", run_cfg)
    write_manifest(out, run_cfg, split="train")
    op = build_operator(cfg, device)
    imgs = torch.stack(training_images(cfg))
    torch.manual_seed(seed)
    net = UNet(channels=imgs.shape[1], base=u.get("base", 48), levels=u.get("levels", 4)).to(device)
    print(f"U-Net params: {n_params(net) / 1e6:.1f}M, {len(imgs)} training images", flush=True)
    opt = torch.optim.Adam(net.parameters(), lr=u.get("lr", 2e-4))
    steps = u.get("steps", 20000)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps)
    scaler = torch.amp.GradScaler("cuda", enabled=(torch.device(device).type == "cuda"))
    g = torch.Generator().manual_seed(seed)
    bs = u.get("batch", 8)
    t0 = time.time()
    ck = out / "checkpoint.pt"
    start = 0
    if ck.exists():  # resume (sessions and containers get restarted)
        c = torch.load(ck, map_location="cpu", weights_only=False)
        net.load_state_dict(c["net"])
        opt.load_state_dict(c["opt"])
        sched.load_state_dict(c["sched"])
        scaler.load_state_dict(c["scaler"])
        g.set_state(c["rng"])
        start = c["step"] + 1
        print(f"resumed from step {start}", flush=True)
    for step in range(start, steps):
        idx = torch.randint(0, len(imgs), (bs,), generator=g)
        x = imgs[idx].to(device)
        if torch.rand(1, generator=g) < 0.5:
            x = x.flip(-1)
        with torch.no_grad():
            y = op.A(x)
            y = y + op.sigma * torch.randn(y.shape, generator=g).to(device)
            inp = op.AT(y, x_shape=x.shape).float()
        with torch.autocast(device_type="cuda", enabled=scaler.is_enabled()):
            loss = ((net(inp) - x) ** 2).mean()
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()
        sched.step()
        if step % 200 == 0 or step == steps - 1:
            print(f"step {step} loss {loss.item():.5f} ({time.time() - t0:.0f}s)", flush=True)
            with atomic_output(ck) as temp, temp.open("wb") as file:
                torch.save({"net": net.state_dict(), "opt": opt.state_dict(), "sched": sched.state_dict(),
                            "scaler": scaler.state_dict(), "rng": g.get_state(), "step": step}, file)
    with atomic_output(out / "unet.pt") as temp, temp.open("wb") as file:
        torch.save({"config": net.config, "state_dict": net.state_dict()}, file)
    print("saved", out / "unet.pt")


if __name__ == "__main__":
    main()
