import os
import sys
import types
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

torch.set_num_threads(min(4, os.cpu_count() or 1))


def dps_repo() -> Path | None:
    """Path to a checkout of the official DPS repo (third_party/dps or $DPS_REPO)."""
    cand = [os.environ.get("DPS_REPO"), ROOT / "third_party" / "dps"]
    for c in cand:
        if c and (Path(c) / "guided_diffusion" / "condition_methods.py").exists():
            return Path(c)
    return None


def import_official_dps():
    repo = dps_repo()
    if repo is None:
        pytest.skip("official DPS repo not available (set DPS_REPO or init the submodule)")
    pytest.importorskip("matplotlib")
    pytest.importorskip("tqdm")
    # util/img_utils.py imports an external clone `motionblur`; stub it (unused here)
    mb = types.ModuleType("motionblur")
    mbm = types.ModuleType("motionblur.motionblur")
    mbm.Kernel = object
    mb.motionblur = mbm
    sys.modules.setdefault("motionblur", mb)
    sys.modules.setdefault("motionblur.motionblur", mbm)
    if str(repo) not in sys.path:
        sys.path.insert(0, str(repo))
    return repo


@pytest.fixture
def tiny_unet():
    from rareprior.diffusion.loader import build_unet
    torch.manual_seed(0)
    m = build_unet(image_size=16, num_channels=32, num_res_blocks=1, learn_sigma=True,
                   attention_resolutions="8", num_heads=1, num_head_channels=16,
                   use_scale_shift_norm=True, resblock_updown=True)
    # random init has zero-initialised output layers; perturb so outputs are non-trivial
    with torch.no_grad():
        for p in m.parameters():
            if p.abs().sum() == 0:
                p.normal_(0, 0.02)
    return m.eval()
