"""
Tests for the per-wavelength integer decimation in ``psdSetToPsfSet``.

``_exact_block_decimate`` point-samples one native sample every k, centred on
the peak (native n_in//2 -> output n_out//2), and rescales to the native flux.
A diffraction-limited PSF (zero PSD) must therefore give the same unit-sum peak
as the legacy shared-grid path at the same output sampling, with the peak and
centroid on pixel n_out//2.
"""

import numpy as np
import pytest

from mastsel.mavisPsf import _exact_block_decimate, psdSetToPsfSet
from mastsel.mavisPsf import defaultArrayBackend as xp

N_PSF = 128


def _to_host(a) -> np.ndarray:
    """Return a numpy array from a numpy or cupy array."""
    return np.asarray(a.get() if hasattr(a, "get") else a)


def _pupil(res: int = 128) -> np.ndarray:
    """Unobstructed circular pupil on a res x res grid."""
    x = np.linspace(-0.5, 0.5, res)
    return (np.hypot(*np.meshgrid(x, x)) <= 0.5).astype(float)


def _centroid(f: np.ndarray) -> tuple:
    """Intensity centroid (row, col) in pixels."""
    yy, xx = np.mgrid[: f.shape[0], : f.shape[1]]
    w = f / f.sum()
    return (w * yy).sum(), (w * xx).sum()


def _per_wavelength_psf(k: int, samp_out: float) -> np.ndarray:
    n = N_PSF * k
    npp = n / (samp_out * k)  # native sampling = samp_out * k
    out = psdSetToPsfSet([[xp.zeros((n, n))]], _pupil(), [1e-6], nPixPup=[npp],
                         freq_range=[npp], dk=[1.0], nPixPsf=N_PSF)
    o = out[0][0] if isinstance(out[0], list) else out[0]
    return _to_host(o.sampling)


def _legacy_psf(ov: int, samp_nat: float) -> np.ndarray:
    n = 256
    npp = n / samp_nat
    out = psdSetToPsfSet([xp.zeros((n, n))], _pupil(), 1e-6, nPixPup=npp,
                         freq_range=npp, dk=1.0, nPixPsf=N_PSF, oversampling=ov)
    return _to_host(out[0].sampling)


# (k, output sampling, legacy oversampling, legacy native sampling)
@pytest.mark.parametrize("k, samp_out, ov, samp_nat",
                         [(2, 1.5, 2, 3.0), (4, 0.5, 2, 1.0), (3, 0.9, 3, 2.7)])
def test_dl_psf_matches_legacy_and_is_centred(k, samp_out, ov, samp_nat):
    f = _per_wavelength_psf(k, samp_out)
    g = _legacy_psf(ov, samp_nat)
    assert f.shape == (N_PSF, N_PSF)
    pk_f, pk_g = f.max() / f.sum(), g.max() / g.sum()
    print(f"k={k} samp={samp_out}: peak {pk_f:.5f} (legacy {pk_g:.5f}), "
          f"argmax {np.unravel_index(f.argmax(), f.shape)}, centroid {_centroid(f)}")
    assert pk_f == pytest.approx(pk_g, rel=1e-3)
    assert np.unravel_index(f.argmax(), f.shape) == (64, 64)
    cy, cx = _centroid(f)
    assert abs(cy - 64) < 0.01 and abs(cx - 64) < 0.01


@pytest.mark.parametrize("k", [2, 3, 4])
@pytest.mark.parametrize("n_out", [8, 9, 16, 17])
def test_decimate_peak_position_and_flux(k, n_out):
    n_in = k * n_out
    a = np.zeros((n_in, n_in))
    a[n_in // 2, n_in // 2] = 5.0
    out = _to_host(_exact_block_decimate(xp.asarray(a), n_out, xp))
    assert out.shape == (n_out, n_out)
    assert np.unravel_index(out.argmax(), out.shape) == (n_out // 2, n_out // 2)
    assert out.sum() == pytest.approx(a.sum(), rel=1e-12)


def test_decimate_k1_returns_input_unchanged():
    a = xp.asarray(np.random.default_rng(0).random((16, 16)))
    assert _exact_block_decimate(a, 16, xp) is a


def test_decimate_non_multiple_raises():
    with pytest.raises(ValueError):
        _exact_block_decimate(xp.ones((10, 10)), 4, xp)


@pytest.mark.parametrize("k", [2, 3, 4])
@pytest.mark.parametrize("n_out", [8, 9])
def test_decimate_is_point_sampling(k, n_out):
    n_in = k * n_out
    x = np.arange(n_in) - n_in // 2
    g = np.exp(-0.5 * (x / (0.4 * n_in)) ** 2)
    a = np.outer(g, g)
    out = _to_host(_exact_block_decimate(xp.asarray(a), n_out, xp))
    idx = n_in // 2 + k * (np.arange(n_out) - n_out // 2)
    ref = a[np.ix_(idx, idx)]
    np.testing.assert_allclose(out / out.max(), ref / ref.max(), rtol=1e-12)
