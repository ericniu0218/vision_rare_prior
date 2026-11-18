import numpy as np
import pytest
import torch

from rareprior.operators.base import adjoint_error
from rareprior.operators.blur import GaussianBlur
from rareprior.operators.lensless import Lensless, prepare_psf
from rareprior.operators.mri import Sense, equispaced_mask, fft2c, ifft2c, variable_density_mask


@pytest.mark.parametrize("boundary", ["reflect", "circular"])
def test_blur_adjoint(boundary):
    op = GaussianBlur(kernel_size=9, std=1.5, boundary=boundary)
    assert adjoint_error(op, (2, 3, 32, 32)) < 1e-10


def test_blur_preserves_constant():
    op = GaussianBlur(kernel_size=61, std=3.0)
    x = torch.full((1, 3, 64, 64), 0.3, dtype=torch.float64)
    assert torch.allclose(op.A(x), x, atol=1e-6)


def test_lensless_adjoint_and_light():
    g = torch.Generator().manual_seed(0)
    psf = torch.rand((1, 24, 24), generator=g, dtype=torch.float64)
    op = Lensless(psf)
    assert adjoint_error(op, (2, 1, 24, 24)) < 1e-10
    # a point source in the centre of the scene puts (almost) all light on the sensor
    x = torch.zeros((1, 1, 24, 24), dtype=torch.float64)
    x[..., 12, 12] = 1
    assert float(op.A(x).sum()) == pytest.approx(1.0, rel=0.3)


def test_prepare_psf_normalised():
    a = np.random.default_rng(0).random((40, 60))
    h = prepare_psf(a, out_shape=(16, 16))
    assert h.shape == (1, 16, 16)
    assert float(h.sum()) == pytest.approx(1.0)
    assert float(h.min()) >= 0


def test_fft_roundtrip_and_sense_adjoint():
    g = torch.Generator().manual_seed(1)
    z = torch.complex(torch.randn(3, 16, 16, generator=g), torch.randn(3, 16, 16, generator=g)).to(torch.complex128)
    assert torch.allclose(ifft2c(fft2c(z)), z, atol=1e-10)
    maps = torch.complex(torch.randn(4, 16, 16, generator=g), torch.randn(4, 16, 16, generator=g))
    maps = maps / maps.abs().pow(2).sum(0).sqrt()
    mask = torch.from_numpy(equispaced_mask((16, 16), 4, calib=4))
    op = Sense(maps, mask, sigma=0.01)
    assert adjoint_error(op, (2, 2, 16, 16)) < 1e-10


def test_variable_density_mask_rate():
    m = variable_density_mask((128, 128), accel=4, calib=16, seed=0)
    assert 3.0 < m.size / m.sum() < 5.0
    assert m[56:72, 56:72].all()


def test_sense_matches_sigpy():
    sp = pytest.importorskip("sigpy")
    import sigpy.mri
    g = np.random.default_rng(0)
    maps = (g.standard_normal((4, 16, 16)) + 1j * g.standard_normal((4, 16, 16)))
    mask = equispaced_mask((16, 16), 2, calib=4)
    x = g.standard_normal((16, 16)) + 1j * g.standard_normal((16, 16))
    ref = sigpy.mri.linop.Sense(maps, weights=mask.astype(float))(x)
    ours = Sense(torch.from_numpy(maps), torch.from_numpy(mask)).A(
        torch.from_numpy(np.stack([x.real, x.imag]))[None])[0].numpy()
    assert np.allclose(ours, ref, atol=1e-8)
    del sp


def test_sense_noise_only_on_sampled_kspace():
    g = torch.Generator().manual_seed(0)
    maps = torch.ones(2, 16, 16, dtype=torch.complex128)
    mask = torch.from_numpy(equispaced_mask((16, 16), 4, calib=2))
    op = Sense(maps, mask, sigma=0.1)
    y = op.add_noise(op.A(torch.zeros(1, 2, 16, 16, dtype=torch.float64)), generator=g)
    assert float(y[..., ~mask].abs().max()) == 0.0
    assert float(y[..., mask].abs().max()) > 0.0


@pytest.mark.parametrize("boundary", ["reflect", "circular"])
def test_blur_fft_equals_direct_conv(boundary):
    g = torch.Generator().manual_seed(0)
    x = torch.randn(2, 3, 80, 72, generator=g, dtype=torch.float64)
    op = GaussianBlur(kernel_size=61, std=3.0, boundary=boundary)
    a = op.A(x)
    op.use_fft = False
    b = op.A(x)
    assert a.shape == b.shape == x.shape
    assert float((a - b).abs().max()) < 1e-10
