"""
Tests for ``pupilOnGrid``: a pupil whose side spans the telescope diameter is
resampled so that the diameter covers ``n_pix_pup`` pixels of a centred grid.
Integer ``n_pix_pup`` keeps the congrid + pad path; non-integer uses linear
affine resampling with the exact scale (no rounding of the diameter).
"""

import numpy as np
import pytest

from mastsel.mavisPsf import (Field, _pad_or_crop_centered, pupilOnGrid,
                              psdSetToPsfSet)
from mastsel.mavisPsf import defaultArrayBackend as xp
from mastsel.mavisUtilities import congrid

EPS = 0.11
N_PUP = 150
N_OUT = 256


def _to_host(a) -> np.ndarray:
    """Return a numpy array from a numpy or cupy array."""
    return np.asarray(a.get() if hasattr(a, "get") else a)


def _disk_pupil(n: int, eps: float = EPS) -> np.ndarray:
    """Binary circular pupil with central obstruction, centred at (n-1)/2."""
    x = np.arange(n) - (n - 1) / 2
    r = np.hypot(*np.meshgrid(x, x))
    return ((r <= n / 2) & (r >= eps * n / 2)).astype(float)


def test_integer_path_identical_to_congrid_pad():
    a = _disk_pupil(N_PUP)
    for n in (64, 70, 71, 100):
        out = _to_host(pupilOnGrid(a, n, N_OUT, xp))
        ref = _to_host(_pad_or_crop_centered(congrid(a, [n, n]), N_OUT, xp))
        assert out.shape == (N_OUT, N_OUT)
        assert np.array_equal(out, ref), n
    # float with integer value takes the integer path too
    out = _to_host(pupilOnGrid(a, 71.0, N_OUT, xp))
    ref = _to_host(_pad_or_crop_centered(congrid(a, [71, 71]), N_OUT, xp))
    assert np.array_equal(out, ref)


def test_non_integer_area_and_centroid():
    a = _disk_pupil(N_PUP)
    n_pix = 70.72
    expected = np.pi / 4 * (1 - EPS**2) * n_pix**2

    out = _to_host(pupilOnGrid(a, n_pix, N_OUT, xp))
    assert out.shape == (N_OUT, N_OUT)
    err = abs(out.sum() / expected - 1)

    err_int = {}
    for n in (70, 71):
        o = _to_host(pupilOnGrid(a, n, N_OUT, xp))
        err_int[n] = abs(o.sum() / expected - 1)
    print(f"area expected={expected:.2f} non-integer={out.sum():.2f} "
          f"(err {100*err:.2f}%), integer 70 err {100*err_int[70]:.2f}%, "
          f"71 err {100*err_int[71]:.2f}%")

    assert err < 0.015
    assert err < min(err_int.values())

    yy, xx = np.indices(out.shape)
    cy = (out * yy).sum() / out.sum()
    cx = (out * xx).sum() / out.sum()
    assert abs(cy - N_OUT // 2) < 0.5
    assert abs(cx - N_OUT // 2) < 0.5


def _dl_psf_unit_sum(n_pix_pup: float, mask: np.ndarray, n: int = N_OUT,
                     diameter: float = 8.0, wvl: float = 0.5e-6) -> np.ndarray:
    """Diffraction-limited PSF (zero PSD) as built in TIPTOP, normalised to unit sum."""
    freq_range = n_pix_pup / diameter  # sx = D * freq_range
    psd_dl = Field(wvl, n, freq_range, 'rad')
    psf = psdSetToPsfSet([psd_dl.sampling], mask, wvl, nPixPup=n_pix_pup,
                         freq_range=freq_range, dk=freq_range / n,
                         nPixPsf=n, oversampling=1)
    s = _to_host(psf[0].sampling)
    return s / s.sum()


def test_dl_peak_scales_with_exact_diameter():
    a = _disk_pupil(N_PUP)
    peaks = {p: _dl_psf_unit_sum(p, a).max() for p in (70, 70.72, 71)}
    print("DL unit-sum peaks:", peaks)

    # unit-sum peak = (sum P)^2 / (sum P^2 N^2) ~ area / N^2 ~ nPixPup^2
    ratio = peaks[70.72] / peaks[70]
    assert ratio == pytest.approx((70.72 / 70) ** 2, rel=5e-3)
    assert peaks[70] < peaks[70.72] < peaks[71]

    # absolute value from the exact (unpixelated) pupil area
    expected = np.pi / 4 * (1 - EPS**2) * 70.72**2 / N_OUT**2
    assert peaks[70.72] == pytest.approx(expected, rel=1e-2)
