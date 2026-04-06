#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import tempfile
import unittest

import numpy as np

from mastsel.mavisLO import cpuArray as lo_cpuArray, resolve_config_path
from mastsel.mavisPsf import (
    FFTConvolve,
    KernelConvolve,
    defaultArrayBackend,
    ft_ft2,
    ft_ift2,
    mastselPsfPrecision,
    psdSetToPsfSet,
)
from mastsel.mavisUtilities import congrid


class TestMastselUtils(unittest.TestCase):
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

    def test_congrid_spline_respects_target_shape(self):
        image = np.arange(9, dtype=np.float32).reshape(3, 3)
        out = congrid(image, (5, 4), method='spline')
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
            np.testing.assert_allclose(np.asarray(restored), image, atol=1e-10)

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
                    self.assertTrue(np.isfinite(np.asarray(sampling)).all())


if __name__ == '__main__':
    unittest.main()
