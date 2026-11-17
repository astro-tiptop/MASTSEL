#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Regression tests for ``maskSA`` with a per-star list of sub-aperture numbers.

Bug: ``saMask`` was allocated once outside the per-star loop. With
``len(nSA) == nMask`` and more than one entry with ``nSA > 1``, the central squares
accumulated in the same array and every such list entry was the same object (the
last modified one). ``maskSA([1, 4, 8], 3, pupil)`` returned masks with sums
``[full pupil, (N/4)^2, (N/4)^2]`` instead of ``[full pupil, (N/4)^2, (N/8)^2]``.

Expected: one independent array per star; for ``nSA >= 4`` the mask is the centered
square of side N/nSA (not multiplied by the pupil).
"""

import unittest

import numpy as np

from mastsel.mavisLO import cpuArray
from mastsel.mavisPsf import maskSA

N = 64  # divisible by 8


def _centered_square(n, side):
    ref = np.zeros((n, n))
    lo = n // 2 - side // 2
    ref[lo:lo + side, lo:lo + side] = 1
    return ref


def _host(a):
    return np.asarray(cpuArray(a))


class TestMaskSA(unittest.TestCase):
    def setUp(self):
        self.pupil = np.ones((N, N))

    def test_list_of_masks_are_independent(self):
        masks = maskSA([1, 4, 8], 3, self.pupil)
        self.assertEqual(len(masks), 3)
        m = [_host(x) for x in masks]
        np.testing.assert_array_equal(m[0], self.pupil)
        np.testing.assert_array_equal(m[1], _centered_square(N, N // 4))
        np.testing.assert_array_equal(m[2], _centered_square(N, N // 8))
        self.assertEqual(m[1].sum(), (N // 4) ** 2)
        self.assertEqual(m[2].sum(), (N // 8) ** 2)
        self.assertIsNot(masks[1], masks[2])
        self.assertFalse(np.shares_memory(m[1], m[2]))

    def test_does_not_modify_pupil(self):
        maskSA([1, 4, 8], 3, self.pupil)
        np.testing.assert_array_equal(self.pupil, np.ones((N, N)))

    def test_order_independence(self):
        a = [_host(x) for x in maskSA([8, 4], 2, self.pupil)]
        b = [_host(x) for x in maskSA([4, 8], 2, self.pupil)]
        np.testing.assert_array_equal(a[0], b[1])
        np.testing.assert_array_equal(a[1], b[0])
        self.assertEqual([x.sum() for x in a], [(N // 8) ** 2, (N // 4) ** 2])

    def test_single_mask_case_unchanged(self):
        mask = maskSA([4], 3, self.pupil)  # len(nSA) != nMask -> single array
        self.assertIsInstance(mask, np.ndarray)
        np.testing.assert_array_equal(mask, _centered_square(N, N // 4))

    def test_equal_entries_have_equal_values(self):
        m0, m1 = [_host(x) for x in maskSA([4, 4], 2, self.pupil)]
        np.testing.assert_array_equal(m0, m1)
        np.testing.assert_array_equal(m0, _centered_square(N, N // 4))


if __name__ == "__main__":
    unittest.main()
