"""
Tests for the unpadded lag-grid path of ``longExposurePsf`` (even N, OTF on the
PSD grid): the N-point IFFT of the PSD times 1/4 must reproduce the 2x-decimated
IFFT of the 2N zero-padded PSD, and the physics (diffraction limit, Marechal)
must be preserved.
"""

import numpy as np
import pytest

from mastsel.mavisPsf import Field, longExposurePsf, zeroPad, ft_ft2
from mastsel.mavisPsf import defaultArrayBackend as xp

WVL = 0.5e-6
N_PUP = 32


def _to_host(a) -> np.ndarray:
    """Return a numpy array from a numpy or cupy array."""
    return np.asarray(a.get() if hasattr(a, "get") else a)


def _mask(n: int) -> Field:
    """Circular pupil of N_PUP pixels on an n x n grid, pixel size 1 (freq_range = 1)."""
    mask = Field(WVL, n, float(n), 'm')
    yy, xx = np.indices((n, n))
    r = np.hypot(xx - n // 2, yy - n // 2)
    mask.sampling = xp.asarray((r <= N_PUP / 2).astype(float))
    return mask


def _psd_field(n: int, shape: np.ndarray) -> Field:
    psd = Field(WVL, n, 1.0, 'rad')
    psd.sampling = xp.asarray(shape)
    return psd


def _frequency_radius(n: int) -> np.ndarray:
    f = (np.arange(n) - n // 2) / n
    fx, fy = np.meshgrid(f, f)
    return np.hypot(fx, fy)


def _von_karman_like(n: int, sigma2: float = 3.0) -> np.ndarray:
    """(k^2 + k0^2)^(-11/6) PSD scaled to a phase variance B(0) = sigma2 (code convention).

    A large sigma2 makes the turbulence OTF differ clearly from 1, so the
    equivalence test is not trivially satisfied.
    """
    k = _frequency_radius(n)
    g = (k**2 + 0.02**2) ** (-11 / 6)
    coeff = (2 * np.pi / WVL) ** 2
    return sigma2 * (2 * n) ** 2 / (coeff * g.sum()) * g


def _bump(n: int, sigma2: float) -> np.ndarray:
    """High-frequency Gaussian ring whose phase variance (code convention) is sigma2.

    sigma2 = B(0) = coeff / M**2 * sum(psd) with M = 2N, coeff = (2 pi / WVL * freq_range)**2.
    """
    g = np.exp(-((_frequency_radius(n) - 0.25) ** 2) / (2 * 0.03**2))
    coeff = (2 * np.pi / WVL) ** 2
    return sigma2 * (2 * n) ** 2 / (coeff * g.sum()) * g


def _old_psf(mask: Field, psd: Field) -> np.ndarray:
    """Reference: 2N zero-padded IFFT, stride-2 decimation around the centre."""
    n = psd.N
    coeff = xp.asarray((psd.kk * psd.width) ** 2)
    padded = zeroPad(psd.sampling, n // 2, xp)
    B = xp.real(xp.fft.ifft2(xp.fft.ifftshift(padded))) * coeff
    b0 = float(B[0, 0])
    B = xp.fft.fftshift(B) - b0
    B = xp.exp(B)
    assert B.shape[0] == 2 * n
    c = B.shape[0] // 2
    idx = c + 2 * xp.arange(-(n // 2), n - n // 2)
    B = B[idx][:, idx]
    maskC = Field(mask.wvl, mask.N, mask.N / psd.width, xp=xp)
    maskC.sampling = xp.copy(mask.sampling)
    maskC.pupilToOtf()
    return _to_host(xp.real(ft_ft2(B * maskC.sampling, xp)))


@pytest.mark.parametrize("n", [64, 128])
def test_unpadded_matches_padded_decimated(n):
    mask = _mask(n)
    psd = _psd_field(n, _von_karman_like(n))
    new = _to_host(longExposurePsf(mask, psd).sampling)
    old = _old_psf(mask, psd)
    err = np.abs(new - old).max() / np.abs(old).max()
    print(f"N={n}: max relative difference new vs old = {err:.2e}")
    assert err < 1e-12


def test_zero_psd_gives_diffraction_limited_psf():
    n = 64
    mask = _mask(n)
    out = _to_host(longExposurePsf(mask, _psd_field(n, np.zeros((n, n)))).sampling)
    m = _to_host(mask.sampling)
    otf = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(np.abs(
        np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(m)))) ** 2)))
    dl = np.real(np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(otf))))
    err = np.abs(out - dl).max() / dl.max()
    print(f"zero PSD vs DL PSF: max relative error = {err:.2e}")
    assert err < 1e-10


def test_marechal_strehl():
    n, sigma2 = 128, 0.1
    mask = _mask(n)
    dl = _to_host(longExposurePsf(mask, _psd_field(n, np.zeros((n, n)))).sampling)
    tb = _to_host(longExposurePsf(mask, _psd_field(n, _bump(n, sigma2))).sampling)
    sr = tb.max() / dl.max()
    print(f"SR = {sr:.5f}, exp(-sigma2) = {np.exp(-sigma2):.5f}")
    assert sr == pytest.approx(np.exp(-sigma2), rel=3e-2)


def test_odd_grid_runs_and_flux_matches_even_neighbour():
    sums = {}
    for n in (128, 129):
        out = _to_host(longExposurePsf(_mask(n), _psd_field(n, _von_karman_like(n))).sampling)
        assert out.shape == (n, n)
        sums[n] = out.sum()
    print(f"PSF sums: {sums}")
    assert sums[129] == pytest.approx(sums[128], rel=1e-2)
