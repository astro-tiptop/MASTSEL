"""
Tests for ``longExposurePsf`` on even and odd grid sizes (exact 2x decimation of
the lag grid) and for the MASTSEL backend selection (MASTSEL_DISABLE_GPU).

Phase PSD convention of ``longExposurePsf``
-------------------------------------------
With M the size of the zero-padded PSD (2N for even N, 2N-1 for odd N) and
``coeff = (kk * freq_range)**2``, the phase autocorrelation is
``B(tau) = coeff * ifft2(ifftshift(psd_padded))`` and the OTF factor is
``exp(B(tau) - B(0))``. Hence the variance entering the exponent is

    sigma2 = B(0) = coeff / M**2 * sum(psd_padded).

The tests build the PSD as ``A * g`` (g a Gaussian bump away from k=0) and pick
A so that sigma2 equals the requested value for the given M. A bump at high
spatial frequency (many cycles across the pupil) makes B(tau) oscillate fast
compared with the telescope OTF, so the turbulence OTF averages to
``exp(-sigma2)`` over the pupil OTF support: SR = peak(turb)/peak(DL) ~ exp(-sigma2).
"""

import os
import subprocess
import sys

import numpy as np
import pytest

from mastsel.mavisPsf import Field, longExposurePsf
from mastsel.mavisPsf import defaultArrayBackend as xp

WVL = 0.5e-6
N_PUP = 32          # pupil diameter in pixels (same physical size for all N)
F_BUMP = 0.25       # bump centre, fraction of the Nyquist-free half range
F_SIG = 0.03        # bump width, fraction of freq_range


def _to_host(a) -> np.ndarray:
    """Return a numpy array from a numpy or cupy array."""
    return np.asarray(a.get() if hasattr(a, "get") else a)


def _setup(n: int, sigma2: float):
    """Build mask and PSD Fields on an n x n grid with pixel size 1 (freq_range = 1).

    Parameters
    ----------
    n : int
        Grid size (even or odd).
    sigma2 : float
        Target B(0) = phase variance in the code's convention (rad^2); 0 gives a zero PSD.
    """
    freq_range = 1.0
    pitch = 1.0 / freq_range
    mask = Field(WVL, n, pitch * n, 'm')
    yy, xx = np.indices((n, n))
    r = np.hypot(xx - n // 2, yy - n // 2)
    mask.sampling = xp.asarray((r <= N_PUP / 2).astype(float))

    psd = Field(WVL, n, freq_range, 'rad')
    f = (np.arange(n) - n // 2) / n * freq_range  # physical frequency of each pixel
    fx, fy = np.meshgrid(f, f)
    g = np.exp(-((np.hypot(fx, fy) - F_BUMP) ** 2) / (2 * F_SIG**2))
    m = 2 * n if n % 2 == 0 else 2 * n - 1
    coeff = (2 * np.pi / WVL * freq_range) ** 2
    amp = sigma2 * m**2 / (coeff * g.sum()) if sigma2 > 0 else 0.0
    psd.sampling = xp.asarray(amp * g)
    return mask, psd


def _psf(n: int, sigma2: float) -> np.ndarray:
    mask, psd = _setup(n, sigma2)
    return _to_host(longExposurePsf(mask, psd).sampling)


def test_zero_psd_equals_dl_psf():
    for n in (128, 129):
        mask, psd = _setup(n, 0.0)
        out = _to_host(longExposurePsf(mask, psd).sampling)
        m = _to_host(mask.sampling)
        # DL PSF from the pupil autocorrelation (FFT of |FFT(P)|^2 centred on n//2)
        otf = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(np.abs(
            np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(m)))) ** 2)))
        dl = np.real(np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(otf))))
        assert out.shape == (n, n)
        assert np.argmax(out) == np.ravel_multi_index((n // 2, n // 2), (n, n))
        assert np.allclose(out / out.sum(), dl / dl.sum(), atol=1e-6 * out.max() / out.sum())


@pytest.mark.parametrize("sigma2", [0.2, 1.0])
def test_parity_strehl_and_flux(sigma2):
    res = {}
    for n in (128, 129):
        dl = _psf(n, 0.0)
        tb = _psf(n, sigma2)
        res[n] = (tb.max() / dl.max(), tb.sum() / dl.sum())
    print(f"sigma2={sigma2}: SR/flux N=128 {res[128]}, N=129 {res[129]}, "
          f"exp(-sigma2)={np.exp(-sigma2):.6f}")
    assert res[128][0] == pytest.approx(res[129][0], abs=1e-3)
    for n in (128, 129):
        assert res[n][1] == pytest.approx(1.0, abs=1e-6)
        # high-frequency bump: SR ~ exp(-sigma2); tolerance covers the residual
        # B(tau) pupil-OTF overlap of the finite-width bump
        assert res[n][0] == pytest.approx(np.exp(-sigma2), abs=5e-3)


def _run_import(env_extra: dict) -> dict:
    code = (
        "import json, numpy, mastsel, mastsel.mavisPsf, mastsel.mavisLO\n"
        "print('RESULT', json.dumps({'gpu': bool(mastsel.gpuEnabled),"
        " 'psf_np': mastsel.mavisPsf.defaultArrayBackend is numpy,"
        " 'lo_np': mastsel.mavisLO.cp is numpy}))\n"
    )
    env = dict(os.environ, PYTHONPATH=os.pathsep.join(sys.path), **env_extra)
    if "MASTSEL_DISABLE_GPU" not in env_extra:
        env.pop("MASTSEL_DISABLE_GPU", None)
    out = subprocess.run([sys.executable, "-c", code], env=env, cwd="/tmp",
                         capture_output=True, text=True, check=True).stdout
    import json
    line = [l for l in out.splitlines() if l.startswith("RESULT ")][-1]
    return json.loads(line[len("RESULT "):])


def test_disable_gpu_env_selects_numpy():
    r = _run_import({"MASTSEL_DISABLE_GPU": "TRUE"})
    assert r == {"gpu": False, "psf_np": True, "lo_np": True}


def test_gpu_enabled_without_env_var():
    pytest.importorskip("cupy")
    assert _run_import({})["gpu"] is True
