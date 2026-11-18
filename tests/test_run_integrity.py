import pandas as pd
import pytest

from rareprior.data.splits import FFHQ, require_development
from rareprior.utils.manifest import atomic_output
from scripts.analyze_survival import check_pairs


def test_draft_rejects_holdout_through_split_or_explicit_id():
    with pytest.raises(ValueError, match="disabled"):
        require_development("test")
    with pytest.raises(ValueError, match="development partition"):
        require_development("dev", [FFHQ["test_pool"][0]], "ffhq_hf")
    require_development("dev", FFHQ["dev"], "ffhq_hf")


def test_atomic_output_does_not_publish_partial_file(tmp_path):
    target = tmp_path / "rows.parquet"
    target.write_text("complete")
    with pytest.raises(RuntimeError), atomic_output(target) as scratch:
        scratch.write_text("partial")
        raise RuntimeError("interrupted")
    assert target.read_text() == "complete"
    assert list(tmp_path.iterdir()) == [target]


def test_pairing_checks_locations_and_severity():
    feature = dict(kind="feature", severity="S1", image_id=0, cy=14, cx=14,
                   cls="blob", size=3, variant=0, sign=1, level=1, amplitude=0.4)
    check_pairs(pd.DataFrame([dict(feature, solver="a"), dict(feature, solver="b")]), "a", "b")
    with pytest.raises(ValueError, match="severities"):
        check_pairs(pd.DataFrame([dict(feature, solver="a"), dict(feature, solver="b", severity="S2")]), "a", "b")


def test_full_fov_identity_psf_reads_central_sensor_crop():
    import torch

    from rareprior.operators.lensless import Lensless

    psf = torch.zeros(8, 8, dtype=torch.float64)
    psf[4, 4] = 1
    camera = Lensless(psf, fov="full")
    scene = torch.zeros(1, 1, 16, 16, dtype=torch.float64)
    scene[0, 0, 5, 9] = 1
    scene[0, 0, 1, 2] = 10
    expected = scene[:, :, 4:12, 4:12]
    assert torch.allclose(camera.A(scene), expected, atol=1e-12)
    footprint = torch.zeros_like(scene)
    footprint[:, :, 4:12, 4:12] = expected
    assert torch.allclose(camera.AT(expected), footprint, atol=1e-12)
