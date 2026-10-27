"""
Tests for ``dirichletResampleMatrix`` and its use in the legacy path of
``psdSetToPsfSet``.

The native PSF (N x N, FFT of a band-limited OTF) is periodic and band limited,
so it is resampled onto the output grid with the exact interpolant
out = G @ A @ G.T, G_ij = D_N((i - n_out//2) r - (j - n_in//2)). Checks:

- Integer ratio: D_N(k) = delta_k0, so the result is centred point sampling
  (and equals ``_exact_block_decimate`` up to the flux normalisation).
- Non-integer ratio: a periodic function with harmonics below Nyquist is
  reproduced at the output points to rounding error.
- Output samples beyond the native field are zero, the others are untouched.
- CPU/GPU parity (GPU only for the matrix; psdSetToPsfSet against a host
  numpy reference built from its own native PSF).
"""

import warnings

import numpy as np
import pytest

import mastsel
from mastsel.mavisLO import cpuArray
from mastsel.mavisPsf import (_exact_block_decimate, dirichletResampleMatrix,
                              psdSetToPsfSet)
from mastsel.mavisPsf import defaultArrayBackend as xp

gpu_only = pytest.mark.skipif(not mastsel.gpuEnabled, reason="GPU backend only")

_PHASES = (0.3, 1.1, 2.0, 0.7)


def _harmonics(N: int) -> list:
    """Harmonics (mx, my, amp, phase), the highest strictly below N/2."""
    m_max = (N - 1) // 2 if N % 2 else N // 2 - 1
    return [(1, 0, 1.0, _PHASES[0]), (2, 3, 0.7, _PHASES[1]),
            (m_max, 1, 0.5, _PHASES[2]), (m_max - 1, m_max, 0.4, _PHASES[3])]


def _band_limited(N: int, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Periodic (period N) band-limited function sampled on x (rows) by y (cols)."""
    f = np.zeros((len(x), len(y)))
    for mx, my, a, ph in _harmonics(N):
        f += a * np.cos(2 * np.pi * (mx * x[:, None] + my * y[None, :]) / N + ph)
    return f


def _resample(A: np.ndarray, n_out: int, r: float) -> np.ndarray:
    """Host result of G @ A @ G.T computed on the default backend."""
    G = dirichletResampleMatrix(A.shape[0], n_out, r, xp)
    return cpuArray(G @ xp.asarray(A) @ G.T)


# ------------------------------------------------------------ integer ratio
@pytest.mark.parametrize("r", [2, 3, 4])
@pytest.mark.parametrize("n_in, n_out", [(48, 12), (60, 15), (49, 12), (61, 15),
                                        (50, 11), (64, 16)])
def test_integer_ratio_is_centred_point_sampling(r, n_in, n_out):
    if (n_out // 2) * r > n_in / 2:
        pytest.skip("output field larger than the native field")
    A = np.random.default_rng(n_in * 10 + r).random((n_in, n_in))
    out = _resample(A, n_out, r)
    idx = n_in // 2 + r * (np.arange(n_out) - n_out // 2)
    np.testing.assert_allclose(out, A[idx][:, idx], rtol=0, atol=1e-12)

    if n_in % n_out == 0 and n_in // n_out == r:
        dec = cpuArray(_exact_block_decimate(xp.asarray(A), n_out, xp))
        np.testing.assert_allclose(out / out.sum(), dec / dec.sum(),
                                   rtol=0, atol=1e-12)


@pytest.mark.parametrize("N", [32, 33])
def test_ratio_one_is_identity(N):
    A = np.random.default_rng(1).random((N, N))
    np.testing.assert_allclose(_resample(A, N, 1.0), A, rtol=0, atol=1e-12)


# --------------------------------------------------------- non-integer ratio
@pytest.mark.parametrize("r", [1.37, 0.61, 2.5])
@pytest.mark.parametrize("N", [64, 65])
def test_non_integer_ratio_reproduces_band_limited_function(N, r):
    n_out = int(0.9 * N / r)
    n_out -= n_out % 2 == 0  # odd output size, also exercises the offset
    assert n_out * r < N
    xj = np.arange(N) - N // 2
    xo = (np.arange(n_out) - n_out // 2) * r
    A = _band_limited(N, xj, xj)
    out = _resample(A, n_out, r)
    ref = _band_limited(N, xo, xo)
    err = np.abs(out - ref).max()
    print(f"N={N} r={r} n_out={n_out}: max err {err:.2e}")
    assert err < 1e-12


@pytest.mark.parametrize("n_out", [20, 21])
def test_even_n_nyquist_harmonic_below_nyquist_is_exact(n_out):
    # single harmonic at N/2 - 1 on an even grid, fine non-integer step
    N, r = 40, 0.77
    xj = np.arange(N) - N // 2
    xo = (np.arange(n_out) - n_out // 2) * r
    m = N // 2 - 1
    A = np.cos(2 * np.pi * m * xj / N + 0.4)[:, None] * np.ones(N)[None, :]
    out = _resample(A, n_out, r)
    ref = np.cos(2 * np.pi * m * xo / N + 0.4)[:, None] * np.ones(n_out)[None, :]
    assert np.abs(out - ref).max() < 1e-12


# ----------------------------------------------------- outside native field
@pytest.mark.parametrize("r", [1.37, 2.5])
@pytest.mark.parametrize("N", [32, 33])
def test_outside_native_field_is_zero(N, r):
    n_out = int(1.6 * N / r)
    assert n_out * r > N
    xj = np.arange(N) - N // 2
    i = np.arange(n_out) - n_out // 2
    xo = i * r
    A = _band_limited(N, xj, xj)
    out = _resample(A, n_out, r)
    inside = np.abs(xo) <= N / 2
    assert inside.any() and (~inside).any()
    assert np.all(out[~inside, :] == 0) and np.all(out[:, ~inside] == 0)
    ref = _band_limited(N, xo[inside], xo[inside])
    np.testing.assert_allclose(out[np.ix_(inside, inside)], ref, rtol=0, atol=1e-12)


# ------------------------------------------------------------ CPU/GPU parity
@gpu_only
@pytest.mark.parametrize("n_in, n_out, r", [(64, 33, 1.37), (65, 40, 0.61),
                                            (64, 16, 4.0), (65, 80, 1.9)])
def test_matrix_gpu_matches_numpy(n_in, n_out, r):
    import cupy
    g_gpu = cpuArray(dirichletResampleMatrix(n_in, n_out, r, cupy))
    g_cpu = dirichletResampleMatrix(n_in, n_out, r, np)
    np.testing.assert_allclose(g_gpu, g_cpu, rtol=0, atol=1e-12)


def _pupil(res: int = 128) -> np.ndarray:
    x = np.linspace(-0.5, 0.5, res)
    return (np.hypot(*np.meshgrid(x, x)) <= 0.5).astype(float)


def test_legacy_multi_wavelength_matches_host_reference():
    """Check the wiring of ``psdSetToPsfSet``, not the kernel.

    The reference is built with ``dirichletResampleMatrix`` itself, so this
    verifies that the legacy path applies G per wavelength with the right ratio,
    size and flux normalisation; the kernel is tested by the other tests here.
    """
    # The backend is fixed at import time (Field and dtypes follow
    # defaultArrayBackend), so numpy and cupy cannot both run psdSetToPsfSet in
    # one process. Reference: the same call with zoom = 1 and N_out = N (no
    # resampling) returns the native PSF; it is resampled here with a numpy G
    # and renormalised to the native flux, i.e. the legacy-path definition.
    n, npp = 128, 64.0
    wvl = [1.0e-6, 1.4e-6, 2.2e-6]
    ov = [1.0, 1.3, 1.7]
    n_pix = 48
    rng = np.random.default_rng(3)
    psds = [xp.asarray(rng.random((n, n)) * 1e-6)]
    kw = dict(nPixPup=npp, freq_range=npp, dk=1.0)
    out = psdSetToPsfSet(psds, _pupil(), wvl, nPixPsf=n_pix, oversampling=ov, **kw)
    native = psdSetToPsfSet(psds, _pupil(), wvl, nPixPsf=n, oversampling=1, **kw)
    assert len(out) == len(wvl)
    for w, o, nat in zip(wvl, out, native):
        # zoom_factor = native_ps / target_ps = w_native / (w_min * ov)
        zoom = (w / wvl[0]) / ov[wvl.index(w)]
        A = cpuArray(nat[0].sampling).astype(np.float64)
        G = cpuArray(dirichletResampleMatrix(n, n_pix, 1.0 / zoom, np))
        ref = G @ A @ G.T
        ref *= A.sum() / ref.sum()
        got = cpuArray(o[0].sampling).astype(np.float64)
        rel = np.abs(got - ref).max() / ref.max()
        print(f"wvl {w}: zoom {zoom:.4f}, max rel diff {rel:.2e}")
        assert rel < 1e-12


# ------------------------------------- integer ratio: indexing == G @ A @ G.T
def _legacy_call(n: int, npp: float, ov: float, n_pix: int):
    """Single-wavelength zero-PSD legacy call; returns the output Field."""
    kw = dict(nPixPup=npp, freq_range=npp, dk=1.0)
    return psdSetToPsfSet([xp.zeros((n, n))], _pupil(), 1e-6, nPixPsf=n_pix,
                          oversampling=ov, **kw)[0]


@pytest.mark.parametrize("ov, n_out", [(2, 32), (2, 33), (3, 40), (3, 41),
                                       (2, 100), (2, 101), (3, 64), (3, 65)])
def test_legacy_integer_ratio_matches_matrix_reference(ov, n_out):
    # Native N = 128, so n_out * ov > N for n_out >= 65 (ov=2) / 43 (ov=3):
    # the zero rows/columns beyond |x| > N/2 are exercised there.
    n, npp = 128, 64.0
    A = cpuArray(_legacy_call(n, npp, 1, n).sampling).astype(np.float64)
    got = cpuArray(_legacy_call(n, npp, ov, n_out).sampling).astype(np.float64)
    G = dirichletResampleMatrix(n, n_out, float(ov), np)
    assert set(np.unique(G)) <= {0.0, 1.0}
    ref = G @ A @ G.T
    ref *= A.sum() / ref.sum()
    rel = np.abs(got - ref).max() / ref.max()
    print(f"ov={ov} n_out={n_out}: max rel diff {rel:.2e}")
    assert got.shape == (n_out, n_out)
    assert rel < 1e-14
    if n_out * ov > n:
        x = ov * (np.arange(n_out) - n_out // 2)
        outside = np.abs(x) > n / 2
        assert outside.any()
        assert np.all(got[outside, :] == 0) and np.all(got[:, outside] == 0)


# ------------------------------------------------- native sampling < 2 warning
def _native_sampling_warnings(n: int, npp: float, ov: float, n_pix: int) -> list:
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        _legacy_call(n, npp, ov, n_pix)
    return [x for x in w if issubclass(x.category, UserWarning)
            and "native sampling" in str(x.message)]


def test_warning_non_integer_ratio_below_native_sampling_2():
    # native sampling = 128 / 96 = 1.33 < 2
    assert len(_native_sampling_warnings(128, 96.0, 1.5, 32)) == 1


@pytest.mark.parametrize("n, npp, ov", [
    (128, 64.0, 1.5),   # non-integer ratio, native sampling = 2 (not below)
    (128, 32.0, 1.5),   # non-integer ratio, native sampling = 4
    (128, 96.0, 2),     # integer ratio, native sampling 1.33: indexing, no G
    (128, 96.0, 1),     # unit ratio, native sampling 1.33
])
def test_no_native_sampling_warning(n, npp, ov):
    assert _native_sampling_warnings(n, npp, ov, 32) == []
