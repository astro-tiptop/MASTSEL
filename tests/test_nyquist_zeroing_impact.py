#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import unittest

import numpy as np

from mastsel.mavisPsf import psdSetToPsfSet

FIXTURE_DIR = os.path.join(os.path.dirname(__file__), "fixtures", "psd")


def _load_fixture_paths():
    if not os.path.isdir(FIXTURE_DIR):
        return []
    return [
        os.path.join(FIXTURE_DIR, name)
        for name in sorted(os.listdir(FIXTURE_DIR))
        if name.endswith(".npz")
    ]


def _flatten_psf_set(psf_set):
    if isinstance(psf_set[0], list):
        return [item.sampling for row in psf_set for item in row]
    return [item.sampling for item in psf_set]


def _peak_rel_diff(psf_a, psf_b):
    a = np.asarray(psf_a, dtype=np.float64)
    b = np.asarray(psf_b, dtype=np.float64)
    a = a / a.sum()
    b = b / b.sum()
    return abs(b.max() - a.max()) / max(a.max(), 1e-20)


def _l2_rel_diff(psf_a, psf_b):
    a = np.asarray(psf_a, dtype=np.float64)
    b = np.asarray(psf_b, dtype=np.float64)
    a = a / a.sum()
    b = b / b.sum()
    denom = np.linalg.norm(a)
    if denom <= 0:
        return 0.0
    return np.linalg.norm(b - a) / denom


def _pad_or_crop_centered(arr, target_n):
    arr = np.asarray(arr)
    n = arr.shape[0]
    if n == target_n:
        return arr
    if target_n < n:
        start = n // 2 - target_n // 2
        return arr[start : start + target_n, start : start + target_n]

    out = np.zeros((target_n, target_n), dtype=arr.dtype)
    start = target_n // 2 - n // 2
    out[start : start + n, start : start + n] = arr
    return out


def _zero_nyquist_row_col(psd):
    out = np.asarray(psd).copy()
    if out.ndim != 2 or out.shape[0] != out.shape[1]:
        raise ValueError("PSD must be a square 2D array")
    if out.shape[0] % 2 == 0:
        # For fftshift-centered even-sized grids, index 0 corresponds to the
        # unique Nyquist line representation.
        out[0, :] = 0.0
        out[:, 0] = 0.0
    return out


def _expand_with_nyquist_row_col(psd):
    """
    Build an odd-sized grid (N+1) by reusing the Nyquist row/column from index 0.
    This is an exploratory transform used only for impact analysis.
    """
    arr = np.asarray(psd)
    if arr.ndim != 2 or arr.shape[0] != arr.shape[1]:
        raise ValueError("PSD must be a square 2D array")
    n = arr.shape[0]
    if n % 2 != 0:
        return arr.copy()

    out = np.zeros((n + 1, n + 1), dtype=arr.dtype)
    out[1:, 1:] = arr
    out[0, 1:] = arr[0, :]
    out[1:, 0] = arr[:, 0]
    out[0, 0] = arr[0, 0]
    return out


@unittest.skipIf(not _load_fixture_paths(), "PSD fixtures not found. Run TIPTOP export script first.")
class TestNyquistZeroingImpact(unittest.TestCase):
    def test_zeroing_row_col_impact_is_measurable_and_finite(self):
        fixture_paths = _load_fixture_paths()
        self.assertGreater(len(fixture_paths), 0)

        processed_even_cases = 0
        for fixture_path in fixture_paths:
            data = np.load(fixture_path, allow_pickle=False)
            case_name = str(data["case_name"])
            N = int(data["N"])
            if N % 2 != 0:
                continue

            processed_even_cases += 1
            input_psds = np.asarray(data["input_psds"])
            zeroed_psds = np.asarray([_zero_nyquist_row_col(psd) for psd in input_psds])

            kwargs = dict(
                mask=np.asarray(data["mask"]),
                wavelength=np.asarray(data["wavelength"]),
                N=N,
                nPixPup=int(data["nPixPup"]),
                grid_diameter=float(data["grid_diameter"]),
                freq_range=float(data["freq_range"]),
                dk=float(data["dk"]),
                nPixPsf=int(data["nPixPsf"]),
                wvlRef=float(data["wvlRef"]),
                oversampling=float(data["oversampling"]),
                padPSD=bool(data["padPSD"]),
            )
            has_opd = bool(data["has_opd"])
            if has_opd:
                kwargs["opdMap"] = np.asarray(data["opd_map"])

            base = psdSetToPsfSet(input_psds, **kwargs)
            zeroed = psdSetToPsfSet(zeroed_psds, **kwargs)

            base_psfs = _flatten_psf_set(base)
            zeroed_psfs = _flatten_psf_set(zeroed)
            self.assertEqual(len(base_psfs), len(zeroed_psfs), msg=f"{case_name}: PSF count mismatch")

            for idx, (psf_base, psf_zeroed) in enumerate(zip(base_psfs, zeroed_psfs)):
                peak_diff = _peak_rel_diff(psf_base, psf_zeroed)
                l2_diff = _l2_rel_diff(psf_base, psf_zeroed)
                self.assertTrue(np.isfinite(peak_diff), msg=f"{case_name} psf#{idx}: non-finite peak diff")
                self.assertTrue(np.isfinite(l2_diff), msg=f"{case_name} psf#{idx}: non-finite l2 diff")

        self.assertGreater(processed_even_cases, 0, "No even-sized fixture cases found")

    def test_expand_with_nyquist_row_col_impact_is_measurable_and_finite(self):
        fixture_paths = _load_fixture_paths()
        self.assertGreater(len(fixture_paths), 0)

        processed_even_cases = 0
        for fixture_path in fixture_paths:
            data = np.load(fixture_path, allow_pickle=False)
            case_name = str(data["case_name"])
            N = int(data["N"])
            if N % 2 != 0:
                continue

            processed_even_cases += 1
            input_psds = np.asarray(data["input_psds"])
            expanded_psds = np.asarray([_expand_with_nyquist_row_col(psd) for psd in input_psds])

            base_kwargs = dict(
                mask=np.asarray(data["mask"]),
                wavelength=np.asarray(data["wavelength"]),
                N=N,
                nPixPup=int(data["nPixPup"]),
                grid_diameter=float(data["grid_diameter"]),
                freq_range=float(data["freq_range"]),
                dk=float(data["dk"]),
                nPixPsf=int(data["nPixPsf"]),
                wvlRef=float(data["wvlRef"]),
                oversampling=float(data["oversampling"]),
                padPSD=bool(data["padPSD"]),
            )
            has_opd = bool(data["has_opd"])
            if has_opd:
                base_kwargs["opdMap"] = np.asarray(data["opd_map"])

            # Baseline run on original even-size inputs.
            base = psdSetToPsfSet(input_psds, **base_kwargs)
            base_psfs = _flatten_psf_set(base)

            # Experimental run on expanded odd-size inputs.
            odd_N = N + 1
            odd_nPixPsf = int(data["nPixPsf"]) + 1
            # Keep pupil sampling independent from PSD-only transform.
            odd_nPixPup = int(data["nPixPup"])
            # Keep PSD step fixed (P3 convention): grid_diameter is unchanged.
            odd_grid = float(data["grid_diameter"])
            odd_freq = odd_N / odd_grid
            odd_mask = np.asarray(data["mask"])

            expanded_kwargs = dict(
                mask=odd_mask,
                wavelength=np.asarray(data["wavelength"]),
                N=odd_N,
                nPixPup=odd_nPixPup,
                grid_diameter=odd_grid,
                freq_range=odd_freq,
                dk=float(data["dk"]),
                nPixPsf=odd_nPixPsf,
                wvlRef=float(data["wvlRef"]),
                oversampling=float(data["oversampling"]),
                padPSD=bool(data["padPSD"]),
            )
            if has_opd:
                expanded_kwargs["opdMap"] = np.asarray(data["opd_map"])

            expanded = psdSetToPsfSet(expanded_psds, **expanded_kwargs)
            expanded_psfs = _flatten_psf_set(expanded)
            self.assertEqual(len(base_psfs), len(expanded_psfs), msg=f"{case_name}: PSF count mismatch")

            for idx, (psf_base, psf_expanded) in enumerate(zip(base_psfs, expanded_psfs)):
                psf_expanded_aligned = _pad_or_crop_centered(psf_expanded, np.asarray(psf_base).shape[0])
                peak_diff = _peak_rel_diff(psf_base, psf_expanded_aligned)
                l2_diff = _l2_rel_diff(psf_base, psf_expanded_aligned)
                self.assertTrue(np.isfinite(peak_diff), msg=f"{case_name} psf#{idx}: non-finite peak diff")
                self.assertTrue(np.isfinite(l2_diff), msg=f"{case_name} psf#{idx}: non-finite l2 diff")

        self.assertGreater(processed_even_cases, 0, "No even-sized fixture cases found")
