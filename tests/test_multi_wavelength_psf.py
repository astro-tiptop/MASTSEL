#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Regression tests for multi-wavelength psdSetToPsfSet.

Covers the case where wavelength is an array and oversampling is a non-integer
float (kRef_float), which previously caused longExposurePsf to return None due
to an incorrect freq_range scaling by (wvl/wvl_min).
"""

import unittest
import numpy as np
from mastsel.mavisPsf import Field, psdSetToPsfSet


def _make_synthetic_inputs(N=128, wvl_ref=1.2e-6, grid_diameter=None, nPixPup=None):
    """Return (psd_array, mask_array, freq_range, dk, grid_diameter, nPixPup)."""
    PSDstep = 8e-3 / (wvl_ref * 206264806 * 3)   # 8 mas target at 1.2 µm, k_=3
    freq_range = N * PSDstep
    if grid_diameter is None:
        grid_diameter = 1.0 / PSDstep
    if nPixPup is None:
        nPixPup = N // 4

    # Flat (non-zero) PSD so the PSF is a valid diffraction-limited-like shape.
    psd = np.ones((N, N), dtype=np.float64) * 1e-4
    # Circular aperture mask.
    mask = np.zeros((N, N), dtype=np.float64)
    cy, cx = N // 2, N // 2
    r = nPixPup // 2
    y, x = np.ogrid[-cy:N - cy, -cx:N - cx]
    mask[x ** 2 + y ** 2 <= r ** 2] = 1.0

    dk = 1.0   # arbitrary normalisation factor
    return psd, mask, freq_range, dk, grid_diameter, nPixPup


class TestMultiWavelengthPsfSetNotNone(unittest.TestCase):
    """
    Regression: psdSetToPsfSet must never return None PSFs for multi-wavelength
    input.  Before the fix, the internal freq_range scaling by (wvl/wvl_min)
    caused the pixel-size consistency check in longExposurePsf to fail and the
    function silently returned None, leading to AttributeError downstream.
    """

    def _run_and_check(self, wavelengths, oversampling, nPixPsf=32):
        N = 128
        wvl_min = float(np.min(wavelengths))
        psd, mask, freq_range, dk, grid_diameter, nPixPup = _make_synthetic_inputs(
            N=N, wvl_ref=wvl_min
        )

        result = psdSetToPsfSet(
            [psd],
            mask,
            wavelengths,
            N,
            nPixPup,
            grid_diameter,
            freq_range,
            dk,
            nPixPsf,
            wvl_min,
            oversampling,
            debug_trace=False,
        )

        # Result is a list of lists (one per wavelength, one per PSD).
        self.assertIsNotNone(result)
        self.assertGreater(len(result), 0)
        for row in result:
            if isinstance(row, list):
                for item in row:
                    self.assertIsNotNone(item, "longExposurePsf returned None")
                    self.assertIsNotNone(item.sampling)
                    self.assertEqual(item.sampling.shape, (nPixPsf, nPixPsf))
            else:
                self.assertIsNotNone(row, "longExposurePsf returned None")
                self.assertIsNotNone(row.sampling)
                self.assertEqual(row.sampling.shape, (nPixPsf, nPixPsf))

    def test_mono_integer_oversampling(self):
        """Mono case with integer oversampling — baseline, must always work."""
        self._run_and_check([1.2e-6], oversampling=3)

    def test_mono_float_oversampling(self):
        """Mono case with float oversampling — should work."""
        self._run_and_check([1.2e-6], oversampling=3.0)

    def test_multi_three_bands_integer_oversampling(self):
        """Multi-λ [1.2, 1.66, 2.2] µm with integer oversampling."""
        self._run_and_check([1.2e-6, 1.66e-6, 2.2e-6], oversampling=3)

    def test_multi_three_bands_float_oversampling(self):
        """
        Multi-λ [1.2, 1.66, 2.2] µm with kRef_float=3.667 (non-integer).
        This is the exact scenario that triggered the None-return bug.
        """
        kRef_float = 2 * (2.2e-6 / 1.2e-6)   # = 3.667
        self._run_and_check([1.2e-6, 1.66e-6, 2.2e-6], oversampling=kRef_float)

    def test_multi_two_bands(self):
        """Multi-λ two bands with float oversampling."""
        kRef_float = 1 * (1.66e-6 / 1.2e-6)   # ≈ 1.383
        self._run_and_check([1.2e-6, 1.66e-6], oversampling=kRef_float)


class TestMultiWavelengthTargetPixelScale(unittest.TestCase):
    """
    The target pixel scale (base_target_ps_rad * oversampling) must be identical
    for all wavelengths and equal to the requested psInMas.
    """

    def test_target_ps_equals_psinmas_for_all_wavelengths(self):
        psInMas = 8.0
        rad2mas = 3600 * 180 * 1000 / np.pi
        wavelengths = np.array([1.2e-6, 1.66e-6, 2.2e-6])
        wvl_min = wavelengths.min()

        # Reproduce the frequencyDomain logic.
        k_ = np.array([3, 2, 2])   # ceil(2 / samp) for each λ
        PSDstep = psInMas / (wavelengths * rad2mas * k_)
        idxPmin = int(np.argmin(PSDstep))   # = 2 (2.2 µm)
        PSDstep_val = PSDstep[idxPmin]
        kRef_float = float(k_[idxPmin]) * (wavelengths[idxPmin] / wvl_min)

        freq_step = PSDstep_val
        base_target_ps_rad = float(wvl_min) * freq_step
        target_ps_mas = base_target_ps_rad * kRef_float * rad2mas

        self.assertAlmostEqual(target_ps_mas, psInMas, places=6,
                               msg=f"target_ps={target_ps_mas:.6f} mas ≠ {psInMas} mas")

    def test_zoom_factors_no_truncation(self):
        """
        zoom_factor = native_ps / target_ps must be ≤ 1 for all λ when kRef_float
        is computed correctly, meaning all native PSFs are finer than target →
        pure downsampling, no upsampling artefacts.
        """
        psInMas = 8.0
        rad2mas = 3600 * 180 * 1000 / np.pi
        wavelengths = np.array([1.2e-6, 1.66e-6, 2.2e-6])
        wvl_min = wavelengths.min()
        k_ = np.array([3, 2, 2])
        PSDstep = psInMas / (wavelengths * rad2mas * k_)
        idxPmin = int(np.argmin(PSDstep))
        PSDstep_val = PSDstep[idxPmin]
        kRef_float = float(k_[idxPmin]) * (wavelengths[idxPmin] / wvl_min)

        freq_step = PSDstep_val
        base_target_ps_rad = float(wvl_min) * freq_step
        target_ps_rad = base_target_ps_rad * kRef_float

        for wvl in wavelengths:
            native_ps_rad = float(wvl) * freq_step
            zoom = native_ps_rad / target_ps_rad
            self.assertLessEqual(zoom, 1.0 + 1e-9,
                                 msg=f"λ={wvl*1e6:.2f}µm: zoom_factor={zoom:.4f} > 1 → upsampling needed")


if __name__ == "__main__":
    unittest.main()
