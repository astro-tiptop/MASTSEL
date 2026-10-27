"""
Tests for the ``pixelResponse`` flag of ``psdSetToPsfSet``.

With ``pixelResponse=True`` the native-grid system OTF is multiplied by the
square-pixel transfer function sinc(p fx) sinc(p fy) (``pixelOtf``), where
p = output pixel / native pixel (k per-wavelength, 1/zoom legacy). After the
point-sampling decimation, each output sample is then the PSF convolved with
the output-pixel box, i.e. the PSF integrated over the pixel. Checks:

- pixelOtf is 1 at DC, separable, symmetric, with its first zero at f = 1/p.
- A diffraction-limited PSF (zero PSD) matches a brute-force pixel integral
  (very fine native grid, block mean of K x K point samples per pixel).
- The opdMap branch (otf_tel) applies the sinc exactly once.
- The legacy path agrees with the per-wavelength path at equal output sampling,
  and with the brute-force pixel integral for a non-integer pixel ratio; its
  exact (Dirichlet) resampling makes the result independent of the native
  sampling.
"""

import numpy as np
import pytest

from mastsel.mavisPsf import pixelOtf, psdSetToPsfSet
from mastsel.mavisPsf import defaultArrayBackend as xp

NP = 32


def _to_host(a) -> np.ndarray:
    """Return a numpy array from a numpy or cupy array."""
    return np.asarray(a.get() if hasattr(a, "get") else a)


def _pupil(res: int = 128) -> np.ndarray:
    """Unobstructed circular pupil on a res x res grid."""
    x = np.linspace(-0.5, 0.5, res)
    return (np.hypot(*np.meshgrid(x, x)) <= 0.5).astype(float)


def _per_wavelength_psf(k: int, samp_out: float, n_pix: int = NP, **kw) -> np.ndarray:
    """Zero-PSD PSF, n_pix output pixels at samp_out (lambda/D per pixel)."""
    n = n_pix * k
    npp = n / (samp_out * k)  # native sampling = samp_out * k
    out = psdSetToPsfSet([[xp.zeros((n, n))]], _pupil(), [1e-6], nPixPup=[npp],
                         freq_range=[npp], dk=[1.0], nPixPsf=n_pix, **kw)
    o = out[0][0] if isinstance(out[0], list) else out[0]
    return _to_host(o.sampling)


def _legacy_psf(ov: int, samp_nat: float, n_pix: int = NP, **kw) -> np.ndarray:
    n = 256
    npp = n / samp_nat
    out = psdSetToPsfSet([xp.zeros((n, n))], _pupil(), 1e-6, nPixPup=npp,
                         freq_range=npp, dk=1.0, nPixPsf=n_pix, oversampling=ov, **kw)
    return _to_host(out[0].sampling)


def _unit_sum(f: np.ndarray) -> np.ndarray:
    return f / f.sum()


# ---------------------------------------------------------------- pixelOtf
@pytest.mark.parametrize("n, r", [(64, 4), (65, 5), (128, 4), (96, 3), (64, 2.5)])
def test_pixel_otf_properties(n, r):
    h = _to_host(pixelOtf(n, r, xp)).astype(float)
    c = n // 2
    assert h.shape == (n, n)
    assert not np.iscomplexobj(h)
    assert h[c, c] == pytest.approx(1.0, abs=1e-6)

    # separable: outer product of its central row and column
    np.testing.assert_allclose(h, np.outer(h[:, c], h[c, :]), atol=1e-6)
    # symmetric about the centre sample (even in lag), also under transpose
    m = c if n % 2 else c - 1  # largest offset available on both sides
    for d in range(1, m + 1):
        assert h[c + d, c] == pytest.approx(h[c - d, c], abs=1e-6)
    np.testing.assert_allclose(h, h.T, atol=1e-7)

    # first zero of the sinc at f = 1/p, i.e. lag offset n/r
    off = n / r
    if float(off).is_integer() and c + int(off) < n:
        o = int(off)
        assert abs(h[c + o, c]) < 1e-6
        assert abs(h[c, c + o]) < 1e-6
        assert abs(h[c - o, c]) < 1e-6 if c - o >= 0 else True


# ------------------------------------------------ brute-force pixel integral
K = 41  # odd: pixel block centred on a native sample


def _brute_force_pixel_psf(samp_out: float) -> np.ndarray:
    # k=1: no decimation, native point samples at sampling samp_out * K
    f = _per_wavelength_psf(1, samp_out * K, n_pix=NP * K)
    assert f.shape == (NP * K, NP * K)
    f = np.roll(f, (K // 2, K // 2), axis=(0, 1))  # block j centred on native j*K
    return f.reshape(NP, K, NP, K).mean(axis=(1, 3))


@pytest.mark.parametrize("samp_out, k", [(0.5, 4), (1.0, 2), (2.0, 1)])
def test_dl_psf_matches_brute_force_pixel_integration(samp_out, k):
    pix = _unit_sum(_per_wavelength_psf(k, samp_out, pixelResponse=True))
    point = _unit_sum(_per_wavelength_psf(k, samp_out))
    brute = _unit_sum(_brute_force_pixel_psf(samp_out))

    err = np.abs(pix - brute).max() / pix.max()
    print(f"samp={samp_out} k={k}: max|pix-brute|/peak = {err:.3e}, "
          f"peak pix {pix.max():.4f}, point {point.max():.4f}, brute {brute.max():.4f}")
    assert pix.shape == (NP, NP)
    assert err < 5e-4
    if samp_out <= 1.0:
        assert pix.max() / point.max() < 0.9


# ------------------------------------------------------------ opdMap branch
def test_opdmap_branch_applies_pixel_response_once():
    k, samp_out = 2, 1.0
    n = NP * k
    npp = n / (samp_out * k)

    def run(opd):
        out = psdSetToPsfSet([[xp.zeros((n, n))]], _pupil(), [1e-6], nPixPup=[npp],
                             freq_range=[npp], dk=[1.0], nPixPsf=NP,
                             opdMap=opd, pixelResponse=True)
        o = out[0][0] if isinstance(out[0], list) else out[0]
        return _unit_sum(_to_host(o.sampling))

    ref = run(None)
    with_opd = run(np.zeros((128, 128)))
    err = np.abs(with_opd - ref).max() / ref.max()
    print(f"opdMap zeros vs None: {err:.3e}")
    assert err < 1e-6


# -------------------------------------------------------------- legacy path
# The unit-sum peak of a small crop depends on the wings, which differ between
# the legacy grid (256 samples, zero-padded crop) and the per-wavelength grid
# (n = NP * k, i.e. only NP / samp_out pupil pixels): the two paths differ by
# ~1.4e-2 (NP=32) even WITHOUT pixel response. These ratios (output/native = 2
# and 3, integer) are point sampling in both paths, so the exact resampling
# leaves them unchanged (identical to the former bilinear values): the difference
# is the pupil/FoV discretisation of the per-wavelength grid and shrinks with NP
# (see test_legacy_pixel_response_converges_to_per_wavelength). The pixel-specific
# check is the pixel/point peak ratio inside each path, which cancels most of
# this (measured 1.4e-4 and 6.7e-3 at NP=32), tolerance 1e-2.
@pytest.mark.parametrize("k, samp_out, ov, samp_nat",
                         [(2, 1.5, 2, 3.0), (3, 0.9, 3, 2.7)])
def test_legacy_pixel_response_matches_per_wavelength(k, samp_out, ov, samp_nat):
    f_pix = _unit_sum(_per_wavelength_psf(k, samp_out, pixelResponse=True))
    f_pt = _unit_sum(_per_wavelength_psf(k, samp_out))
    g_pix = _unit_sum(_legacy_psf(ov, samp_nat, pixelResponse=True))
    g_pt = _unit_sum(_legacy_psf(ov, samp_nat))
    ratio_f, ratio_g = f_pix.max() / f_pt.max(), g_pix.max() / g_pt.max()
    print(f"k={k} samp={samp_out}: pix/point peak {ratio_f:.5f} (legacy {ratio_g:.5f}); "
          f"unit-sum peak {f_pix.max():.5f} (legacy {g_pix.max():.5f})")
    assert g_pix.shape == (NP, NP)
    assert ratio_f < 0.9 and ratio_g < 0.9
    # a missing or doubled sinc would move the ratio by >10%
    assert ratio_g == pytest.approx(ratio_f, rel=1e-2)
    assert g_pix.max() == pytest.approx(f_pix.max(), rel=2e-2)  # measured 1.4e-2
    assert np.unravel_index(g_pix.argmax(), g_pix.shape) == (NP // 2, NP // 2)
    assert np.unravel_index(f_pix.argmax(), f_pix.shape) == (NP // 2, NP // 2)


# With NP = 128 the per-wavelength pupil has 4x more pixels and the two paths
# agree much better (measured: ratio 0 and 1.6e-3, unit-sum peak 0 and 1.3e-3).
@pytest.mark.parametrize("k, samp_out, ov, samp_nat",
                         [(2, 1.5, 2, 3.0), (3, 0.9, 3, 2.7)])
def test_legacy_pixel_response_converges_to_per_wavelength(k, samp_out, ov, samp_nat):
    n_pix = 128
    f_pix = _unit_sum(_per_wavelength_psf(k, samp_out, n_pix=n_pix, pixelResponse=True))
    f_pt = _unit_sum(_per_wavelength_psf(k, samp_out, n_pix=n_pix))
    g_pix = _unit_sum(_legacy_psf(ov, samp_nat, n_pix=n_pix, pixelResponse=True))
    g_pt = _unit_sum(_legacy_psf(ov, samp_nat, n_pix=n_pix))
    ratio_f, ratio_g = f_pix.max() / f_pt.max(), g_pix.max() / g_pt.max()
    print(f"NP={n_pix} k={k}: ratio rel {abs(ratio_g / ratio_f - 1):.2e}, "
          f"peak rel {abs(g_pix.max() / f_pix.max() - 1):.2e}")
    assert ratio_g == pytest.approx(ratio_f, rel=5e-3)
    assert g_pix.max() == pytest.approx(f_pix.max(), rel=4e-3)



# Non-integer native/output ratio (P3 kRef_float). The legacy path resamples
# the native PSF with the exact periodic band-limited (Dirichlet) interpolant,
# so the result no longer depends on the native sampling: the same pupil and
# output pixel on a 2x finer native grid give the same PSF to rounding error
# (the former bilinear interpolation differed by 1.6e-2 here, 4.5e-3 without
# pixel response). That invariance is the sharp check of the resampling.
#
# Against the brute-force pixel integral the residual (measured 8.8e-3 of the
# peak, identical on CPU and GPU) is NOT interpolation: it is the pupil
# discretisation of the reference. The reference grid spans NP output pixels, so
# its pupil has only NP / samp_out = 42 pixels, resampled bilinearly from the
# 128 px mask; the legacy pupil has n / samp_nat = 102 pixels. The legacy PSF
# itself is stable to 5e-4 when its own pupil sampling is doubled or quadrupled,
# while the reference error is non-monotonic in NP (8.1e-3 at NP=32, 3.1e-4 at
# 64, 2.3e-3 at 128), the signature of a pupil-edge effect; the brute-force
# block size K matters at the 1e-4 to 6e-4 level only. Threshold unchanged
# (2e-2, 2.3x the measured value).
def _legacy_non_integer(pix: bool, ov: float = 6.6, samp_nat: float = 5.0,
                        n: int = 512) -> np.ndarray:
    npp = n / samp_nat
    out = psdSetToPsfSet([xp.zeros((n, n))], _pupil(), 1e-6, nPixPup=npp,
                         freq_range=npp, dk=1.0, nPixPsf=NP, oversampling=ov,
                         pixelResponse=pix)
    return _unit_sum(_to_host(out[0].sampling))


def test_legacy_non_integer_ratio_matches_brute_force():
    ov, samp_nat = 6.6, 5.0
    samp_out = samp_nat / ov
    brute = _unit_sum(_brute_force_pixel_psf(samp_out))
    err_pix = np.abs(_legacy_non_integer(True) - brute).max() / brute.max()
    err_pt = np.abs(_legacy_non_integer(False) - brute).max() / brute.max()
    print(f"ov={ov} samp_out={samp_out:.3f}: legacy pix vs brute {err_pix:.3e}, "
          f"legacy point vs brute pix {err_pt:.3e}")
    assert err_pix < 2e-2
    assert err_pt > 0.1  # the pixel effect itself is ~0.26: the test discriminates


@pytest.mark.parametrize("pix", [True, False])
def test_legacy_non_integer_ratio_independent_of_native_sampling(pix):
    # same pupil sampling (n / samp_nat = 102.4), same output pixel (ov / samp_nat
    # = 1.32), native grid and sampling doubled: exact resampling gives the same PSF
    a = _legacy_non_integer(pix, ov=6.6, samp_nat=5.0, n=512)
    b = _legacy_non_integer(pix, ov=13.2, samp_nat=10.0, n=1024)
    err = np.abs(a - b).max() / a.max()
    print(f"pixelResponse={pix}: 5/512 vs 10/1024 max rel diff {err:.2e}")
    assert err < 1e-10
