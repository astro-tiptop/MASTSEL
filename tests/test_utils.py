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
    mastselPsfPrecision,
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


if __name__ == '__main__':
    unittest.main()
