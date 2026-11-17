"""
Tests for ``WindowRadiusWCoG = 'optimize'`` in ``MavisLO.computeBiasAndVariance``.

Physical expectation
--------------------
For a Gaussian spot the WCoG with window radius R has an optical gain (bias)
b(R) and a centroid variance var(R) [pixel^2]. With ``LoopGain_LO = 'optimize'``
the bias enters the residuals only through sigma2Noise = var/b^2
(``computeNoiseResidual``, ``computeWindResidual``), and the optimised-gain
residual does not decrease when the noise level increases. Hence the radius
that minimises var/b^2 must also minimise the LO residual nr + wr, and:

1. with a numeric window the result is identical to the previous implementation;
2. the chosen R is the brute-force argmin over R = 1..mediumGridSize//2 of var/b^2;
3. the chosen R minimises nr + wr among the other radii;
4. the chosen R does not decrease with the spot FWHM (as long as it does not
   saturate at the grid edge, mediumGridSize//2 = 25: a known limit, beyond
   which the window cannot grow any further);
5. the per-star radii stored by computeTotalResidualMatrix match the ones of
   the separate computeBiasAndVariance calls.
"""

import configparser
import os
import unittest

import numpy as np

from mastsel.mavisLO import MavisLO, parse_config_value
from mastsel.mavisLO import intRebin, simple2Dgaussian

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
INI_PATH = os.path.join(ROOT, "data", "ini", "mavisParamsTests.ini")
WIND_PSD = os.path.join(ROOT, "data", "windpsd_mavis.fits")

PIXEL_SCALE = 7.5      # [mas], ANDES-like LO detector pixel
FREQ = 500.0           # [Hz]
FLUX = 2.5e5           # [ph/s] per sub-aperture
EE = 0.5               # encircled energy used to scale the spot


def _load_config() -> dict:
    """Test ini as nested dict, with an ANDES-like LO sensor and optimised window/gain."""
    parser = configparser.ConfigParser()
    parser.optionxform = str
    parser.read(INI_PATH)
    cfg = {sec: {k: parse_config_value(v) for k, v in parser[sec].items()}
           for sec in parser.sections()}
    cfg["COMPUTATION"] = {"platform": "CPU"}
    cfg["sensor_LO"]["PixelScale"] = PIXEL_SCALE
    cfg["sensor_LO"]["WindowRadiusWCoG"] = parse_config_value("'optimize'")
    cfg["RTC"]["LoopGain_LO"] = parse_config_value("'optimize'")
    cfg["telescope"]["windPsdFile"] = WIND_PSD
    return cfg


def _old_bias_and_variance(lo, aNGS_flux, aNGS_freq, aNGS_EE, aNGS_FWHM_mas,
                           PixelScale, WindowRadiusWCoG):
    """REFERENCE: body of ``MavisLO.computeBiasAndVariance`` as in origin/main (doLO=True).

    Copied verbatim except for the window radius, which is passed explicitly
    (the old 'optimize' rule, max(ceil(FWHM/2/PixelScale), 1), is not needed here).
    """
    np_ = np
    lo.mediumPixelScale = PixelScale/lo.downsample_factor
    lo.smallGridSize = 2*WindowRadiusWCoG

    aNGS_frameflux = aNGS_flux / aNGS_freq
    asigma = aNGS_FWHM_mas/2.3548200450309493/lo.mediumPixelScale
    g2d = simple2Dgaussian(lo.xLargeGrid, lo.yLargeGrid, 0, 0, asigma)
    g2d = g2d * 1 / np_.sum(g2d)
    I_k_data = g2d * aNGS_EE * aNGS_frameflux
    g2d_prime = simple2Dgaussian(lo.xLargeGrid, lo.yLargeGrid, lo.p_offset, 0, asigma)
    g2d_prime = g2d_prime * 1 / np_.sum(g2d_prime)
    I_k_prime_data = g2d_prime * aNGS_EE * aNGS_frameflux

    I_k_data = intRebin(I_k_data, lo.mediumShape) * lo.downsample_factor**2
    I_k_prime_data = intRebin(I_k_prime_data, lo.mediumShape) * lo.downsample_factor**2
    ffx = np_.arange(-lo.mediumGridSize/2, lo.mediumGridSize/2, 1.0) + 0.5
    (fx, fy) = np_.meshgrid(ffx, ffx)
    W_Mask = np_.where(np_.logical_or(fx**2 + fy**2 > WindowRadiusWCoG**2, fx**2 + fy**2 < 0**2),
                       0.0, 1.0)
    if lo.smallGridSize < lo.mediumGridSize/2:
        ii1, ii2 = int(lo.mediumGridSize/2-lo.smallGridSize), \
                   int(lo.mediumGridSize/2+lo.smallGridSize)
        I_k_data = I_k_data[ii1:ii2, ii1:ii2]
        I_k_prime_data = I_k_prime_data[ii1:ii2, ii1:ii2]
        W_Mask = W_Mask[ii1:ii2, ii1:ii2]
        fx = fx[ii1:ii2, ii1:ii2]
        fy = fy[ii1:ii2, ii1:ii2]
    mu_ktr_array, var_ktr_array, _ = lo.meanVarSigma(I_k_data, doLO=True)
    mu_ktr_prime_array, _, _ = lo.meanVarSigma(I_k_prime_data, doLO=True)
    masked_mu0 = W_Mask * mu_ktr_array
    masked_mu = W_Mask * mu_ktr_prime_array
    masked_sigma = W_Mask**2 * var_ktr_array
    mux = np_.sum(masked_mu*fx)/np_.sum(masked_mu)
    muy = np_.sum(masked_mu*fy)/np_.sum(masked_mu)
    varx = np_.sum(masked_sigma*fx**2)/(np_.sum(masked_mu0)**2)
    vary = np_.sum(masked_sigma*fy**2)/(np_.sum(masked_mu0)**2)
    bias = mux/(lo.p_offset/lo.downsample_factor)
    return (bias, (mux, muy), (varx, vary))


class TestWcogWindowOptimize(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.lo = MavisLO(config_dict=_load_config())
        cls.lo.configLOFreq(FREQ)
        cls.rmax = cls.lo.mediumGridSize // 2

    # ------------------------------------------------------------ helpers

    def _fixed(self, fwhm_px: float, radius: int):
        """computeBiasAndVariance with a numeric window radius (restores the config)."""
        lo = self.lo
        old = lo.WindowRadiusWCoG_LO
        lo.WindowRadiusWCoG_LO = radius
        try:
            return lo.computeBiasAndVariance(FLUX, FREQ, EE, fwhm_px*PIXEL_SCALE, PIXEL_SCALE)
        finally:
            lo.WindowRadiusWCoG_LO = old

    def _optimized(self, fwhm_px: float):
        """computeBiasAndVariance with WindowRadiusWCoG='optimize'; returns (result, R)."""
        self.assertEqual(self.lo.WindowRadiusWCoG_LO, 0)
        res = self.lo.computeBiasAndVariance(FLUX, FREQ, EE, fwhm_px*PIXEL_SCALE, PIXEL_SCALE)
        return res, self.lo.windowRadiusWCoG

    def _noise_equiv_curve(self, fwhm_px: float) -> np.ndarray:
        """var/bias^2 for R = 1..rmax (inf where bias <= 0)."""
        out = []
        for r in range(1, self.rmax + 1):
            bias, _, avar = self._fixed(fwhm_px, r)
            out.append(avar[0]/bias**2 if bias > 0 else np.inf)
        return np.array(out)

    def _residual(self, fwhm_px: float, radius: int) -> float:
        """nr + wr [nm^2] for a fixed radius, as in computeTotalResidualMatrix."""
        lo = self.lo
        bias, _, avar = self._fixed(fwhm_px, radius)
        var1x = avar[0] * PIXEL_SCALE**2 / lo.N_sa_tot_LO[0] * lo.mas2nm**2
        nr = lo.computeNoiseResidual(0.25, lo.maxLOtFreq, int(4*lo.maxLOtFreq), var1x, bias)
        wr = lo.computeWindResidual(lo.psd_freq, lo.psd_tip_wind, lo.psd_tilt_wind, var1x, bias)
        return float(np.sum(nr) + np.sum(wr))

    # -------------------------------------------------------------- tests

    def test_1_numeric_window_unchanged(self):
        for fwhm_px in (2.0, 8.0):
            for radius in (1, 3, 10):
                new = self._fixed(fwhm_px, radius)
                self.assertEqual(self.lo.windowRadiusWCoG, radius)
                ref = _old_bias_and_variance(self.lo, FLUX, FREQ, EE, fwhm_px*PIXEL_SCALE,
                                             PIXEL_SCALE, radius)
                np.testing.assert_allclose(new[0], ref[0], rtol=1e-12, atol=0)
                np.testing.assert_allclose(new[1], ref[1], rtol=1e-12, atol=1e-14)
                np.testing.assert_allclose(new[2], ref[2], rtol=1e-12, atol=0)

    def test_2_optimize_is_bruteforce_argmin(self):
        for fwhm_px in (1.0, 3.0, 10.0, 25.0):
            curve = self._noise_equiv_curve(fwhm_px)
            expected = int(np.argmin(curve)) + 1
            res, radius = self._optimized(fwhm_px)
            self.assertEqual(radius, expected)
            # the returned values are those of the chosen radius
            ref = self._fixed(fwhm_px, radius)
            np.testing.assert_allclose(res[0], ref[0], rtol=1e-12)
            np.testing.assert_allclose(res[2], ref[2], rtol=1e-12)

    def test_3_optimize_minimises_residual(self):
        for fwhm_px in (3.0, 10.0, 20.0):
            _, radius = self._optimized(fwhm_px)
            self.assertLess(radius, self.rmax)
            radii = sorted({1, 2, radius-2, radius-1, radius, radius+1, radius+2,
                            self.rmax} & set(range(1, self.rmax + 1)))
            res = {r: self._residual(fwhm_px, r) for r in radii}
            self.assertLessEqual(res[radius], min(res.values())*(1 + 1e-6))

    def test_4_radius_nondecreasing_with_fwhm(self):
        fwhms = np.arange(1.0, 31.0, 1.0)
        radii = [self._optimized(f)[1] for f in fwhms]
        # saturation at the grid edge (R = rmax) is a known limit: check the
        # monotonic trend only on the non-saturated part
        unsat = [r for r in radii if r < self.rmax]
        self.assertGreater(len(unsat), 5)
        self.assertTrue(all(b >= a for a, b in zip(unsat[:-1], unsat[1:])), unsat)

    def test_5_per_star_storage(self):
        lo = self.lo
        fwhm_px = [2.0, 12.0]
        coords = np.array([[10.0, 0.0], [0.0, 15.0]])
        pointings = np.array([[0.0, 0.0]])
        lo.computeTotalResidualMatrix(pointings, coords,
                                      np.array([FLUX, FLUX]), np.array([FREQ, FREQ]),
                                      np.array([0.3, 0.3]), np.array([EE, EE]),
                                      [f*PIXEL_SCALE for f in fwhm_px])
        self.assertEqual(len(lo.wcogRadius), 2)
        expected = []
        for f in fwhm_px:
            lo.configLOFreq(FREQ)
            self._optimized(f)
            expected.append(lo.windowRadiusWCoG)
        self.assertEqual(list(lo.wcogRadius), expected)
        self.assertNotEqual(expected[0], expected[1])


if __name__ == "__main__":
    unittest.main()
