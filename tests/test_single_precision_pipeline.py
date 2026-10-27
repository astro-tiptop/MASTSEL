"""
Tests for the single-precision pipeline of ``psdSetToPsfSet``.

With ``mastselPsfPrecision('single')`` the pupil mask, the static phase and the
PSD are cast to the model dtype, so ``longExposurePsf`` runs in float32 /
complex64 and the PSFs are float32 (callers pass float64 arrays). In double
precision the output is float64. Single and double must agree to float32
round-off: the PSF is the IFFT of the OTF, so the absolute error has a floor
of a few 1e-9 of the peak in the wings, while the per-pixel relative error is
large only where the PSF is tiny.
"""

import os

import numpy as np
import pytest

import mastsel.mavisPsf as mp
from mastsel.mavisPsf import psdSetToPsfSet

FIXTURE_DIR = os.path.join(os.path.dirname(__file__), "fixtures", "psd")

TOL_MAX_OVER_PEAK = 5e-6
TOL_PEAK_REL = 5e-6
# Largest per-pixel error in the wings, from the float32 FFT round-off (the PSF
# is the IFFT of the OTF): measured <= 1.6e-8 of the peak with numpy and
# <= 1.2e-7 with cuFFT on the fixtures (median per annulus ~1e-9).
TOL_WINGS_OVER_PEAK = 5e-7
WING_THRESHOLD = 1e-3  # wings: PSF < WING_THRESHOLD * peak


def _to_host(a) -> np.ndarray:
    """Return a numpy array from a numpy or cupy array."""
    return np.asarray(a.get() if hasattr(a, "get") else a)


def _flat_psfs(out) -> list:
    """Flatten the psdSetToPsfSet return value into a list of Field objects."""
    flat = []
    for item in out:
        flat.extend(item if isinstance(item, list) else [item])
    return flat


def _run_precision(precision: str, call, monkeypatch) -> tuple:
    """
    Run ``call()`` in the given precision and return host PSFs and native dtypes.

    Parameters
    ----------
    precision : str
        'single' or 'double'.
    call : callable
        Zero-argument function returning the psdSetToPsfSet output.
    monkeypatch : pytest.MonkeyPatch
        Used to spy on ``longExposurePsf`` (native PSF dtype).

    Returns
    -------
    tuple
        (list of host PSF arrays, set of native dtypes returned by longExposurePsf)
    """
    native = set()
    orig = mp.longExposurePsf

    def spy(*args, **kwargs):
        res = orig(*args, **kwargs)
        native.add(np.dtype(_to_host(res.sampling).dtype))
        return res

    mp.mastselPsfPrecision(precision)
    try:
        # context: undo the spy after each run so the next run wraps the original
        with monkeypatch.context() as m:
            m.setattr(mp, "longExposurePsf", spy)
            out = call()
        psfs = [_to_host(f.sampling).copy() for f in _flat_psfs(out)]
    finally:
        mp.mastselPsfPrecision("double")
    return psfs, native


def _metrics(single: np.ndarray, double: np.ndarray) -> tuple:
    """Return (max|d|/peak, relative peak diff, wings max|d|/peak)."""
    s, d = single.astype(np.float64), double.astype(np.float64)
    peak = d.max()
    diff = np.abs(s - d)
    wings = d < WING_THRESHOLD * peak
    return diff.max() / peak, abs(s.max() - peak) / peak, diff[wings].max() / peak


def _compare(call, monkeypatch, label: str) -> None:
    """Run double then single, check dtypes and agreement."""
    dbl, nat_d = _run_precision("double", call, monkeypatch)
    sgl, nat_s = _run_precision("single", call, monkeypatch)
    assert nat_d == {np.dtype(np.float64)}
    assert nat_s == {np.dtype(np.float32)}
    assert len(dbl) == len(sgl) > 0
    for i, (s, d) in enumerate(zip(sgl, dbl)):
        assert d.dtype == np.float64 and s.dtype == np.float32
        assert s.shape == d.shape
        m, p, w = _metrics(s, d)
        print(f"{label} psf#{i}: max/peak={m:.2e} peak_rel={p:.2e} wings/peak={w:.2e}")
        assert m < TOL_MAX_OVER_PEAK
        assert p < TOL_PEAK_REL
        assert w < TOL_WINGS_OVER_PEAK


def _load_fixture(name: str) -> dict:
    path = os.path.join(FIXTURE_DIR, f"{name}.npz")
    if not os.path.exists(path):
        pytest.skip(f"fixture {name} not found")
    return dict(np.load(path, allow_pickle=False))


@pytest.mark.parametrize("oversampling", [1, 2])
@pytest.mark.parametrize("case", ["SOUL", "HARMONI_SCAO"])
def test_legacy_path_single_matches_double(case, oversampling, monkeypatch):
    d = _load_fixture(case)

    def call():
        return psdSetToPsfSet(
            inputPSDs=d["input_psds"], mask=d["mask"], wavelength=d["wavelength"],
            nPixPup=int(d["nPixPup"]), freq_range=float(d["freq_range"]),
            dk=float(d["dk"]), nPixPsf=int(d["nPixPsf"]), oversampling=oversampling,
        )

    _compare(call, monkeypatch, f"{case} ov={oversampling}")


def _pupil(res: int) -> np.ndarray:
    """Unobstructed circular pupil on a res x res grid."""
    x = np.linspace(-0.5, 0.5, res)
    return (np.hypot(*np.meshgrid(x, x)) <= 0.5).astype(float)


def _power_law_psd(n: int, amp: float) -> np.ndarray:
    """Smooth, centred, non-zero PSD ~ (1 + k^2)^(-11/6) with zero at the origin."""
    k = np.arange(n) - n // 2
    kk2 = k[:, None] ** 2 + k[None, :] ** 2
    psd = amp * (1.0 + kk2) ** (-11.0 / 6.0)
    psd[n // 2, n // 2] = 0.0
    return psd


@pytest.mark.parametrize("nonzero", [True, False])
def test_per_wavelength_path_single_matches_double(nonzero, monkeypatch):
    n_psf, k, npp = 64, 2, 48
    n = n_psf * k
    wvls = [1.2e-6, 2.2e-6]
    # amplitude chosen so that, with dk=1e4, the peak drops to ~85% of diffraction limit
    psd = _power_law_psd(n, 1e-5) if nonzero else np.zeros((n, n))

    def call():
        return psdSetToPsfSet([[psd.copy()] for _ in wvls], _pupil(n), wvls,
                              nPixPup=[npp] * 2, freq_range=[float(npp)] * 2,
                              dk=[1e4] * 2, nPixPsf=n_psf)

    _compare(call, monkeypatch, f"per-wvl nonzero={nonzero}")


def test_opdmap_branch_single_matches_double(monkeypatch):
    d = _load_fixture("HARMONI_SCAO")
    if not bool(d["has_opd"]):
        pytest.skip("fixture has no opd map")

    def call():
        return psdSetToPsfSet(
            inputPSDs=d["input_psds"], mask=d["mask"], wavelength=d["wavelength"],
            nPixPup=int(d["nPixPup"]), freq_range=float(d["freq_range"]),
            dk=float(d["dk"]), nPixPsf=int(d["nPixPsf"]), oversampling=1,
            opdMap=d["opd_map"],
        )

    _compare(call, monkeypatch, "HARMONI opdMap")
