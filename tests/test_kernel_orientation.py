"""
Tests for ``residualToSpectrum``: the Fourier-domain Gaussian kernel built from
the residual ellipse [theta, sigma_max, sigma_min] must, once applied with
``convolve``, give an image-domain Gaussian whose major axis is at ``theta``
(counterclockwise from +x = column axis towards +y = row axis) and whose
sigmas are the requested ones. Regression for the mirrored angle (-theta).
"""

import numpy as np
import pytest

from mastsel.mavisPsf import Field, convolve, residualToSpectrum
from mastsel.mavisPsf import defaultArrayBackend as xp

N = 256
PS = 2.0  # mas / pixel
SIGMA_MAX = 12.0  # mas
SIGMA_MIN = 4.0  # mas
WVL = 500e-9


def _to_host(a) -> np.ndarray:
    """Return a numpy array from a numpy or cupy array."""
    return np.asarray(a.get() if hasattr(a, "get") else a)


def _moments(theta_deg: float):
    """Convolve a centred delta with the kernel; return image, angle, sigmas.

    Returns
    -------
    img : numpy.ndarray
        Convolved image.
    angle_deg : float
        Major-axis angle in degrees, counterclockwise from +x (columns).
    sigmas_mas : tuple of float
        Major and minor sigmas in mas.
    """
    delta = Field(WVL, N, N * PS, xp=xp)
    delta.sampling[N // 2, N // 2] = 1.0
    kernel = residualToSpectrum([np.deg2rad(theta_deg), SIGMA_MAX, SIGMA_MIN],
                                WVL, N, 1.0 / (N * PS))
    out = convolve(delta, kernel)
    img = _to_host(out.sampling).astype(float)
    w = img / img.sum()
    r, c = np.mgrid[:N, :N]
    mr, mc = (w * r).sum(), (w * c).sum()
    cov_rr = (w * (r - mr) ** 2).sum()
    cov_cc = (w * (c - mc) ** 2).sum()
    cov_rc = (w * (r - mr) * (c - mc)).sum()
    angle = 0.5 * np.arctan2(2 * cov_rc, cov_cc - cov_rr)
    ev = np.linalg.eigvalsh(np.array([[cov_cc, cov_rc], [cov_rc, cov_rr]]))
    sig = (np.sqrt(ev[1]) * PS, np.sqrt(ev[0]) * PS)
    return img, np.rad2deg(angle), sig


@pytest.mark.parametrize("theta_deg", [0, 30, 45, -30, 60, 90])
def test_kernel_orientation_and_sigmas(theta_deg):
    img, angle, (s_max, s_min) = _moments(theta_deg)
    print(f"theta={theta_deg:4d} measured angle={angle:8.3f} deg "
          f"sigmas={s_max:.3f}, {s_min:.3f} mas")
    diff = (angle - theta_deg + 90.0) % 180.0 - 90.0  # modulo 180 deg
    assert abs(diff) < 0.5
    assert s_max == pytest.approx(SIGMA_MAX, rel=0.01)
    assert s_min == pytest.approx(SIGMA_MIN, rel=0.01)


@pytest.mark.parametrize("theta_deg", [0, 30, -30])
def test_kernel_preserves_flux(theta_deg):
    img, _, _ = _moments(theta_deg)
    assert abs(img.sum() - 1.0) < 1e-6
