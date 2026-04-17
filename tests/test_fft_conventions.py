#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import unittest

import numpy as np

from mastsel.mavisLO import cpuArray as lo_cpuArray
from mastsel.mavisPsf import centeredPixelCoords, defaultArrayBackend, ft_ft2, ft_ift2


class TestFftConventions(unittest.TestCase):
    def test_constant_field_maps_dc_to_centered_pixel(self):
        for n in (32, 33):
            image = np.ones((n, n), dtype=np.float64)
            transformed = ft_ft2(defaultArrayBackend.asarray(image), xp=defaultArrayBackend)
            transformed = np.asarray(lo_cpuArray(transformed))

            cy, cx = centeredPixelCoords(n)
            dc = transformed[cy, cx]

            self.assertAlmostEqual(float(np.real(dc)), float(n * n), places=8)
            self.assertAlmostEqual(float(np.imag(dc)), 0.0, places=8)

            off_center = transformed.copy()
            off_center[cy, cx] = 0.0
            self.assertLess(np.max(np.abs(off_center)), 1e-8)

    def test_centered_delta_has_zero_phase_fft(self):
        for n in (32, 33):
            image = np.zeros((n, n), dtype=np.float64)
            cy, cx = centeredPixelCoords(n)
            image[cy, cx] = 1.0

            transformed = ft_ft2(defaultArrayBackend.asarray(image), xp=defaultArrayBackend)
            transformed = np.asarray(lo_cpuArray(transformed))

            # If a hidden half-pixel shift were present, phase would alternate by pi.
            self.assertLess(np.max(np.abs(np.imag(transformed))), 1e-10)
            self.assertLess(np.max(np.abs(np.real(transformed) - 1.0)), 1e-10)

    def test_centered_frequency_impulse_ifft_is_uniform(self):
        for n in (32, 33):
            spectrum = np.zeros((n, n), dtype=np.complex128)
            cy, cx = centeredPixelCoords(n)
            spectrum[cy, cx] = 1.0 + 0.0j

            image = ft_ift2(defaultArrayBackend.asarray(spectrum), xp=defaultArrayBackend)
            image = np.asarray(lo_cpuArray(image))

            expected = np.full((n, n), 1.0 / (n * n), dtype=np.float64)
            np.testing.assert_allclose(np.real(image), expected, atol=1e-12, rtol=0.0)
            np.testing.assert_allclose(np.imag(image), 0.0, atol=1e-12, rtol=0.0)


if __name__ == '__main__':
    unittest.main()
