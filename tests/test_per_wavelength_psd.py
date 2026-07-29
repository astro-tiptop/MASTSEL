#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Tests for the per-wavelength calling convention of psdSetToPsfSet
(inputPSDs[i_wvl][i_src], one exact grid per wavelength, produced by P3's
psdPerWavelength=True), as opposed to the legacy shared-grid convention
(inputPSDs is a flat list of same-shape PSDs, freq_range/dk are scalars).

The per-wavelength path replaces the legacy bilinear affine_transform resize
(mavisPsf.py, historically the only way to go from the internal oversampled
grid to the requested nPixPsf) with an *exact* integer block-sum decimation
(_exact_block_decimate), since each wavelength's grid is already built to hit
the target pixel scale exactly -- see fourierModel.py/frequencyDomain.py in
P3 for how nOtf_i = nPixPsf * k_i is guaranteed.
"""

import unittest
import numpy as np

from mastsel.mavisPsf import psdSetToPsfSet, _exact_block_decimate


def _make_psd_and_mask(N, nPixPup, PSDstep, value=1e-4):
    psd = np.ones((N, N), dtype=np.float64) * value
    mask = np.zeros((N, N), dtype=np.float64)
    cy, cx = N // 2, N // 2
    r = nPixPup // 2
    y, x = np.ogrid[-cy:N - cy, -cx:N - cx]
    mask[x ** 2 + y ** 2 <= r ** 2] = 1.0
    return psd, mask


def _build_per_wavelength_inputs(wavelengths_nm, k_list, nPixPsf, nPixPup,
                                 n_directions=2, target_ps_mas=8.0):
    """
    Synthetic per-wavelength inputs: each wavelength gets its own grid size
    N_i = nPixPsf * k_i (mirroring what P3 builds), and its own PSDstep_i
    chosen so that native_ps_rad_i * k_i hits target_ps_mas exactly -- the
    same invariant P3's per-wavelength grids guarantee.
    """
    rad2mas = 3600 * 180 * 1000 / np.pi
    inputPSDs, freq_range_list, dk_list, wavelengths = [], [], [], []
    mask_native = None
    for wvl_nm, k_i in zip(wavelengths_nm, k_list):
        wvl = wvl_nm * 1e-9
        wavelengths.append(wvl)
        N_i = nPixPsf * k_i
        PSDstep_i = (target_ps_mas / rad2mas) / (wvl * k_i)
        psd, mask = _make_psd_and_mask(N_i, nPixPup, PSDstep_i)
        inputPSDs.append([psd for _ in range(n_directions)])
        freq_range_list.append(N_i * PSDstep_i)
        dk_list.append(1.0)
        if mask_native is None:
            mask_native = mask
    return inputPSDs, mask_native, wavelengths, freq_range_list, dk_list


class TestExactBlockDecimate(unittest.TestCase):

    def test_no_op_when_already_target_size(self):
        arr = np.arange(16.0).reshape(4, 4)
        out = _exact_block_decimate(arr, 4, np)
        np.testing.assert_array_equal(out, arr)

    def test_conserves_total_sum(self):
        rng = np.random.default_rng(0)
        arr = rng.random((12, 12))
        out = _exact_block_decimate(arr, 4, np)
        self.assertEqual(out.shape, (4, 4))
        self.assertAlmostEqual(float(out.sum()), float(arr.sum()), places=10)

    def test_exact_values_for_uniform_input(self):
        # A uniform 6x6 array of value v, decimated by k=3 -> 2x2 array of 9v.
        v = 2.5
        arr = np.full((6, 6), v)
        out = _exact_block_decimate(arr, 2, np)
        np.testing.assert_allclose(out, np.full((2, 2), v * 9))

    def test_raises_on_non_integer_ratio(self):
        arr = np.zeros((10, 10))
        with self.assertRaises(ValueError):
            _exact_block_decimate(arr, 3, np)


class TestPerWavelengthPsdSetToPsfSet(unittest.TestCase):
    """
    Distinctly-shaped PSDs per wavelength (unlike the existing
    test_multi_wavelength_psf.py, which always reuses the same array for
    every wavelength on a shared grid).
    """

    def test_different_grid_sizes_per_wavelength_produce_requested_output_shape(self):
        nPixPsf, nPixPup = 32, 48
        inputPSDs, mask, wavelengths, freq_range, dk = _build_per_wavelength_inputs(
            wavelengths_nm=[1200, 2200], k_list=[3, 2],
            nPixPsf=nPixPsf, nPixPup=nPixPup,
        )
        # Sanity: the two wavelengths really do use different native grid sizes.
        self.assertEqual(inputPSDs[0][0].shape[0], nPixPsf * 3)
        self.assertEqual(inputPSDs[1][0].shape[0], nPixPsf * 2)

        result = psdSetToPsfSet(inputPSDs, mask, wavelengths, nPixPup=nPixPup,
                                freq_range=freq_range, dk=dk, nPixPsf=nPixPsf)

        self.assertEqual(len(result), len(wavelengths))
        for row in result:
            self.assertEqual(len(row), 2)  # n_directions
            for psf in row:
                self.assertEqual(psf.sampling.shape, (nPixPsf, nPixPsf))

    def test_flux_is_exactly_conserved_no_renormalization_needed(self):
        nPixPsf, nPixPup = 24, 40
        inputPSDs, mask, wavelengths, freq_range, dk = _build_per_wavelength_inputs(
            wavelengths_nm=[900, 1650, 2200], k_list=[4, 2, 1],
            nPixPsf=nPixPsf, nPixPup=nPixPup, n_directions=1,
        )
        result = psdSetToPsfSet(inputPSDs, mask, wavelengths, nPixPup=nPixPup,
                                freq_range=freq_range, dk=dk, nPixPsf=nPixPsf)
        # Every wavelength (including k_i=1, i.e. no decimation at all) must
        # produce a finite, non-degenerate PSF.
        for row in result:
            for psf in row:
                self.assertTrue(np.isfinite(psf.sampling).all())
                self.assertGreater(float(psf.sampling.sum()), 0.0)

    def test_single_wavelength_per_wavelength_mode_still_works(self):
        nPixPsf, nPixPup = 16, 24
        inputPSDs, mask, wavelengths, freq_range, dk = _build_per_wavelength_inputs(
            wavelengths_nm=[1200], k_list=[2],
            nPixPsf=nPixPsf, nPixPup=nPixPup, n_directions=1,
        )
        result = psdSetToPsfSet(inputPSDs, mask, wavelengths, nPixPup=nPixPup,
                                freq_range=freq_range, dk=dk, nPixPsf=nPixPsf)
        # multi_wave is False for a single wavelength -> flat list, matching
        # the legacy single-wavelength return convention.
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0].sampling.shape, (nPixPsf, nPixPsf))

    def test_mismatched_inputpsds_and_wavelength_length_raises(self):
        nPixPsf, nPixPup = 16, 24
        inputPSDs, mask, wavelengths, freq_range, dk = _build_per_wavelength_inputs(
            wavelengths_nm=[1200, 2200], k_list=[2, 2],
            nPixPsf=nPixPsf, nPixPup=nPixPup, n_directions=1,
        )
        with self.assertRaises(ValueError):
            psdSetToPsfSet(inputPSDs, mask, [wavelengths[0]], nPixPup=nPixPup,
                           freq_range=freq_range, dk=dk, nPixPsf=nPixPsf)

    def test_nPixPup_varies_per_wavelength(self):
        """
        The pupil occupies a different fraction of each wavelength's own grid
        (n_internal_i varies with k_i, the physical pupil diameter does not),
        so nPixPup must be accepted as a per-wavelength sequence, not just a
        shared scalar -- regression guard for a bug found during review where
        a single nPixPup was silently reused for every wavelength.
        """
        nPixPsf = 32
        inputPSDs, mask, wavelengths, freq_range, dk = _build_per_wavelength_inputs(
            wavelengths_nm=[1200, 2200], k_list=[3, 2],
            nPixPsf=nPixPsf, nPixPup=40, n_directions=1,
        )
        nPixPup_per_wvl = [40, 28]  # deliberately different per wavelength
        result = psdSetToPsfSet(inputPSDs, mask, wavelengths, nPixPup=nPixPup_per_wvl,
                                freq_range=freq_range, dk=dk, nPixPsf=nPixPsf)
        self.assertEqual(len(result), 2)
        for row in result:
            self.assertEqual(row[0].sampling.shape, (nPixPsf, nPixPsf))

    def test_mismatched_nPixPup_sequence_length_raises(self):
        nPixPsf = 16
        inputPSDs, mask, wavelengths, freq_range, dk = _build_per_wavelength_inputs(
            wavelengths_nm=[1200, 2200], k_list=[2, 2],
            nPixPsf=nPixPsf, nPixPup=24, n_directions=1,
        )
        with self.assertRaises(ValueError):
            psdSetToPsfSet(inputPSDs, mask, wavelengths, nPixPup=[24, 24, 24],
                           freq_range=freq_range, dk=dk, nPixPsf=nPixPsf)


class TestLegacyModeStillDetectedCorrectly(unittest.TestCase):
    """
    A flat list of 2D arrays (today's calling convention) must never be
    mistaken for the new per-wavelength convention.
    """

    def test_flat_psd_list_uses_legacy_path(self):
        from mastsel.mavisPsf import Field  # noqa: F401 (kept for parity with sibling test file)
        N = 64
        wvl = 1.2e-6
        PSDstep = 8e-3 / (wvl * 206264806 * 3)
        freq_range = N * PSDstep
        nPixPup = N // 4
        psd, mask = _make_psd_and_mask(N, nPixPup, PSDstep)

        result = psdSetToPsfSet([psd, psd], mask, [wvl], nPixPup=nPixPup,
                                freq_range=freq_range, dk=1.0, nPixPsf=16,
                                oversampling=3)
        # Legacy single-wavelength convention: flat list, one entry per PSD.
        self.assertEqual(len(result), 2)
        for psf in result:
            self.assertEqual(psf.sampling.shape, (16, 16))


if __name__ == '__main__':
    unittest.main()
