"""Build and load ADM U-Nets (guided-diffusion architecture).

Unlike the official DPS ``create_model``, which prints a warning and silently
falls back to *random weights* if loading fails, this loader raises. It also
checks the checkpoint's SHA-256 when one is given.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import torch

from .guided.unet import UNetModel

# Configs copied from the official DPS repo (configs/model_config.yaml and
# configs/imagenet_model_config.yaml), commit effbde7.
FFHQ_CONFIG = dict(image_size=256, num_channels=128, num_res_blocks=1, channel_mult="",
                   learn_sigma=True, class_cond=False, attention_resolutions="16", num_heads=4,
                   num_head_channels=64, num_heads_upsample=-1, use_scale_shift_norm=True,
                   dropout=0.0, resblock_updown=True, use_fp16=False, use_new_attention_order=False)
IMAGENET_CONFIG = dict(FFHQ_CONFIG, num_channels=256, num_res_blocks=2, attention_resolutions="32,16,8")

FFHQ_SHA256 = "81d535743156ec6be34d8668e6920da94f0614074d7793a16c8fa9e306237faa"
HF_SOURCES = {
    "ffhq": ("deepinv/diffunet", "diffusion_ffhq_10m.pt"),
    "ffhq_alt": ("galleonli/ffhq_10m", "ffhq_10m.pt"),
    "imagenet": ("deepinv/diffunet", "diffusion_openai.pt"),
}


def build_unet(image_size, num_channels, num_res_blocks, channel_mult="", learn_sigma=False,
               class_cond=False, attention_resolutions="16", num_heads=1, num_head_channels=-1,
               num_heads_upsample=-1, use_scale_shift_norm=False, dropout=0.0, resblock_updown=False,
               use_fp16=False, use_new_attention_order=False, in_channels=3, use_checkpoint=False):
    if channel_mult == "":
        channel_mult = {512: (0.5, 1, 1, 2, 2, 4, 4), 256: (1, 1, 2, 2, 4, 4), 128: (1, 1, 2, 3, 4),
                        64: (1, 2, 3, 4), 32: (1, 2, 2), 16: (1, 2)}[image_size]
    elif isinstance(channel_mult, str):
        channel_mult = tuple(int(c) for c in channel_mult.split(","))
    if isinstance(attention_resolutions, str):
        attention_ds = tuple(image_size // int(r) for r in attention_resolutions.split(",") if r)
    else:
        attention_ds = tuple(image_size // int(r) for r in attention_resolutions)
    return UNetModel(
        image_size=image_size, in_channels=in_channels, model_channels=num_channels,
        out_channels=(in_channels if not learn_sigma else 2 * in_channels),
        num_res_blocks=num_res_blocks, attention_resolutions=attention_ds, dropout=dropout,
        channel_mult=channel_mult, num_classes=(1000 if class_cond else None),
        use_checkpoint=use_checkpoint, use_fp16=use_fp16, num_heads=num_heads,
        num_head_channels=num_head_channels, num_heads_upsample=num_heads_upsample,
        use_scale_shift_norm=use_scale_shift_norm, resblock_updown=resblock_updown,
        use_new_attention_order=use_new_attention_order)


def sha256_file(path, chunk=1 << 22) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while b := f.read(chunk):
            h.update(b)
    return h.hexdigest()


DEEPINV_REVISION = "6b0a2cf45decab1f022d22eefe3787d9d515f32b"


def download_checkpoint(name: str, cache_dir: str | None = None) -> Path:
    from huggingface_hub import hf_hub_download
    repo, fname = HF_SOURCES[name]
    revision = DEEPINV_REVISION if repo == "deepinv/diffunet" else None
    return Path(hf_hub_download(repo_id=repo, filename=fname, cache_dir=cache_dir, revision=revision))


def load_adm(name: str = "ffhq", path: str | None = None, device="cpu", check_sha: bool = True):
    cfg = FFHQ_CONFIG if name.startswith("ffhq") else IMAGENET_CONFIG
    path = Path(path) if path else download_checkpoint(name)
    if check_sha and name.startswith("ffhq"):
        digest = sha256_file(path)
        if digest != FFHQ_SHA256:
            raise ValueError(f"checkpoint SHA-256 mismatch: {digest}")
    model = build_unet(**cfg)
    state = torch.load(path, map_location="cpu", weights_only=True)
    model.load_state_dict(state, strict=True)
    return model.to(device).eval()
