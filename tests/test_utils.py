#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import tempfile
import unittest

import numpy as np

from mastsel.mavisLO import cpuArray as lo_cpuArray, resolve_config_path
from mastsel.mavisPsf import (
    FFTConvolve,
    KernelConvolve,
    _expand_odd_psd_to_even_with_zero_nyquist,
    defaultArrayBackend,
    ft_ft2,
    ft_ift2,
    mastselPsfPrecision,
    psdSetToPsfSet,
)
from mastsel.mavisUtilities import congrid


class TestMastselUtils(unittest.TestCase):
    def test_expand_odd_psd_to_even_with_zero_nyquist_values(self):
        odd = np.array(
            [
                [1.0, 2.0, 3.0],
                [4.0, 5.0, 6.0],
                [7.0, 8.0, 9.0],
            ],
            dtype=np.float64,
        )
        converted = _expand_odd_psd_to_even_with_zero_nyquist(odd, xp=defaultArrayBackend)
        converted = np.asarray(lo_cpuArray(converted))

        expected = np.array(
            [
                [0.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 2.0, 3.0],
                [0.0, 4.0, 5.0, 6.0],
                [0.0, 7.0, 8.0, 9.0],
            ],
            dtype=np.float64,
        )
        np.testing.assert_allclose(converted, expected, atol=0.0, rtol=0.0)

    def test_expand_odd_psd_to_even_with_zero_nyquist_even_passthrough(self):
        even = np.arange(16, dtype=np.float64).reshape(4, 4)
        converted = _expand_odd_psd_to_even_with_zero_nyquist(even, xp=defaultArrayBackend)
        converted = np.asarray(lo_cpuArray(converted))
        np.testing.assert_allclose(converted, even, atol=0.0, rtol=0.0)

    def test_cpu_array_accepts_python_containers(self):
        values = [1, 2, 3]
        self.assertEqual(lo_cpuArray(values), values)

    def test_fft_convolve_empty_inputs(self):
        empty = defaultArrayBackend.asarray([])
        out = FFTConvolve(empty, empty, xp=defaultArrayBackend)
        self.assertEqual(out.size, 0)

    def test_kernel_convolve_empty_inputs(self):
        empty = defaultArrayBackend.asarray([])
        out = KernelConvolve(empty, empty, xp=defaultArrayBackend)
        self.assertEqual(out.size, 0)

    def test_congrid_respects_target_shape(self):
        image = np.arange(9, dtype=np.float32).reshape(3, 3)
        out = congrid(image, (5, 4))
        self.assertEqual(out.shape, (5, 4))
        self.assertEqual(out.dtype, image.dtype)

    def test_mastsel_psf_precision_dtype_switch(self):
        mastselPsfPrecision(dtype=np.float32)
        mastselPsfPrecision(dtype=np.dtype(np.float32))
        mastselPsfPrecision(dtype=np.float64)
        mastselPsfPrecision(dtype=np.dtype(np.float64))

    def test_resolve_config_path_prefers_existing_path_root_file(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            target = f"{tmpdir}/config.ini"
            with open(target, 'w', encoding='utf-8') as handle:
                handle.write('[section]\nvalue = 1\n')

            resolved = resolve_config_path('config.ini', tmpdir, '/unused/p3', '/unused/tiptop')
            self.assertEqual(resolved, target)

    def test_fourier_roundtrip_supports_even_and_odd_grids(self):
        for n in (32, 33):
            image = np.zeros((n, n), dtype=np.float64)
            image[n // 2, n // 2] = 1.0
            transformed = ft_ft2(defaultArrayBackend.asarray(image), xp=defaultArrayBackend)
            restored = ft_ift2(transformed, xp=defaultArrayBackend)
            np.testing.assert_allclose(np.asarray(lo_cpuArray(restored)), image, atol=1e-10)

    def test_psd_set_to_psf_set_handles_even_and_odd_psd_shapes(self):
        cases = [
            (1.65e-6, 1.65e-6, False),
            ([1.65e-6, 2.2e-6], 2.2e-6, True),
        ]

        for n in (32, 33):
            n_pix_pup = n - 4
            grid_diameter = 8.0
            freq_range = n / grid_diameter
            mask = np.ones((n_pix_pup, n_pix_pup), dtype=np.float64)
            psd = np.zeros((n, n), dtype=np.float64)

            for wavelength, wvl_ref, pad_psd in cases:
                out = psdSetToPsfSet(
                    [psd],
                    mask,
                    wavelength,
                    n,
                    n_pix_pup,
                    grid_diameter,
                    freq_range,
                    1.0,
                    16,
                    wvl_ref,
                    2,
                    padPSD=pad_psd,
                )

                if pad_psd:
                    samplings = [item.sampling for row in out for item in row]
                else:
                    samplings = [item.sampling for item in out]

                for sampling in samplings:
                    self.assertEqual(sampling.shape, (16, 16))
                    self.assertTrue(np.isfinite(np.asarray(lo_cpuArray(sampling))).all())

    def test_psd_set_to_psf_set_internal_grid_mode_defaults_to_even_legacy(self):
        n = 32
        n_pix_pup = 28
        grid_diameter = 8.0
        freq_range = n / grid_diameter
        mask = np.ones((n_pix_pup, n_pix_pup), dtype=np.float64)
        psd = np.zeros((n, n), dtype=np.float64)

        out = psdSetToPsfSet(
            [psd],
            mask,
            1.65e-6,
            n,
            n_pix_pup,
            grid_diameter,
            freq_range,
            1.0,
            16,
            1.65e-6,
            1,
        )

        self.assertEqual(out[0].sampling.shape, (16, 16))

    def test_psd_set_to_psf_set_odd_internal_raises_on_parity_mismatch_without_oversampling(self):
        n = 32
        requested_npix = 16  # even on purpose
        n_pix_pup = 28
        grid_diameter = 8.0
        freq_range = n / grid_diameter
        mask = np.ones((n_pix_pup, n_pix_pup), dtype=np.float64)
        psd = np.zeros((n, n), dtype=np.float64)

        with self.assertRaisesRegex(ValueError, 'nPixPsf parity must match PSD parity'):
            psdSetToPsfSet(
                [psd],
                mask,
                1.65e-6,
                n,
                n_pix_pup,
                grid_diameter,
                freq_range,
                1.0,
                requested_npix,
                1.65e-6,
                1,
                internal_grid_mode='odd_internal',
            )

    def test_psd_set_to_psf_set_even_legacy_accepts_odd_input_grid(self):
        n = 33
        requested_npix = 33
        n_pix_pup = 29
        grid_diameter = 8.0
        freq_range = n / grid_diameter
        mask = np.ones((n_pix_pup, n_pix_pup), dtype=np.float64)
        psd = np.zeros((n, n), dtype=np.float64)

        out = psdSetToPsfSet(
            [psd],
            mask,
            1.65e-6,
            n,
            n_pix_pup,
            grid_diameter,
            freq_range,
            1.0,
            requested_npix,
            1.65e-6,
            1,
            internal_grid_mode='even_legacy',
        )

        self.assertEqual(out[0].sampling.shape, (requested_npix, requested_npix))

    def test_psd_set_to_psf_set_even_legacy_raises_on_parity_mismatch_without_oversampling(self):
        n = 33
        requested_npix = 32  # mismatch on purpose
        n_pix_pup = 29
        grid_diameter = 8.0
        freq_range = n / grid_diameter
        mask = np.ones((n_pix_pup, n_pix_pup), dtype=np.float64)
        psd = np.zeros((n, n), dtype=np.float64)

        with self.assertRaisesRegex(ValueError, 'nPixPsf parity must match PSD parity'):
            psdSetToPsfSet(
                [psd],
                mask,
                1.65e-6,
                n,
                n_pix_pup,
                grid_diameter,
                freq_range,
                1.0,
                requested_npix,
                1.65e-6,
                1,
                internal_grid_mode='even_legacy',
            )

    def test_psd_set_to_psf_set_odd_internal_accepts_odd_input_grid(self):
        n = 33
        requested_npix = 33
        n_pix_pup = 29
        grid_diameter = 8.0
        freq_range = n / grid_diameter
        mask = np.ones((n_pix_pup, n_pix_pup), dtype=np.float64)
        psd = np.zeros((n, n), dtype=np.float64)

        out = psdSetToPsfSet(
            [psd],
            mask,
            1.65e-6,
            n,
            n_pix_pup,
            grid_diameter,
            freq_range,
            1.0,
            requested_npix,
            1.65e-6,
            1,
            internal_grid_mode='odd_internal',
        )

        self.assertEqual(out[0].sampling.shape, (requested_npix, requested_npix))

    def test_psd_set_to_psf_set_odd_internal_keeps_requested_size_with_oversampling(self):
        n = 32
        requested_npix = 16
        n_pix_pup = 28
        grid_diameter = 8.0
        freq_range = n / grid_diameter
        mask = np.ones((n_pix_pup, n_pix_pup), dtype=np.float64)
        psd = np.zeros((n, n), dtype=np.float64)

        out = psdSetToPsfSet(
            [psd],
            mask,
            1.65e-6,
            n,
            n_pix_pup,
            grid_diameter,
            freq_range,
            1.0,
            requested_npix,
            1.65e-6,
            2,
            internal_grid_mode='odd_internal',
        )

        self.assertEqual(out[0].sampling.shape, (requested_npix, requested_npix))


if __name__ == '__main__':
    unittest.main()
