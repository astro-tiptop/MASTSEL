#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import unittest

import numpy as np

from mastsel.mavisLO import cpuArray as lo_cpuArray
from mastsel.mavisPsf import Field, congrid, psdSetToPsfSet, zeroPad

FIXTURE_DIR = os.path.join(os.path.dirname(__file__), "fixtures", "psd")

# Tolerances for peak differences between old vs new PSF generation.
PEAK_TOL = 1.0e-3

# Per-case odd/even tolerances derived from
# tests/report_nyquist_zeroing_impact.py (mode=expand_to_n_plus_1),
# converted from percent to fractional units.
ODD_EVEN_PEAK_TOL_BY_CASE = {
    "ERIS": 0.624960 / 100.0,
    "HARMONI_SCAO": 1.536845 / 100.0,
    "MAVIS": 4.959511 / 100.0,
    "SOUL": 0.466357 / 100.0,
    "SPHERE": 0.401884 / 100.0,
}
# Small absolute safety margin for numerical drift across environments.
ODD_EVEN_TOL_MARGIN = 1.0e-3


def _ft_ft2_old(G):
    return np.fft.fftshift(np.fft.fft2(np.fft.fftshift(G)))


def _load_fixture_paths():
    if not os.path.isdir(FIXTURE_DIR):
        return []
    return [
        os.path.join(FIXTURE_DIR, name)
        for name in sorted(os.listdir(FIXTURE_DIR))
        if name.endswith(".npz")
    ]


def _pad_or_crop_centered(arr, target_n):
    arr = np.asarray(lo_cpuArray(arr))
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


def _expand_with_nyquist_row_col(psd):
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


def _peak_rel_diff(psf_a, psf_b):
    a = np.asarray(lo_cpuArray(psf_a)).astype(np.float64, copy=False)
    b = np.asarray(lo_cpuArray(psf_b)).astype(np.float64, copy=False)
    a = a / a.sum()
    b = b / b.sum()
    return abs(b.max() - a.max()) / max(a.max(), 1e-20)


def _odd_even_peak_tol(case_name):
    default_tol = max(ODD_EVEN_PEAK_TOL_BY_CASE.values())
    return ODD_EVEN_PEAK_TOL_BY_CASE.get(case_name, default_tol) + ODD_EVEN_TOL_MARGIN


def _psdset_old_reference(
    input_psds,
    mask,
    wavelength,
    N,
    nPixPup,
    grid_diameter,
    freq_range,
    dk,
    nPixPsf,
    wvlRef,
    oversampling,
    opd_map,
    has_opd,
    padPSD,
):
    wavelength = np.atleast_1d(wavelength)
    oversampling = np.atleast_1d(oversampling)
    if len(oversampling) == 1:
        oversampling = np.full_like(wavelength, oversampling[0], dtype=np.float64)

    out = []
    for wvl, ovrsmp in zip(wavelength, oversampling):
        if padPSD:
            pixRatio = wvlRef / wvl
            oRatio = np.ceil(pixRatio) / pixRatio
            ovrsmp *= np.ceil(pixRatio)
            ovrsmp = int(ovrsmp)
            nPad = int(np.round(N * oRatio / 2) * 2)
        else:
            nPad = int(N)

        maskField = Field(wvl, nPad, grid_diameter)
        maskField.sampling = congrid(mask, [nPixPup, nPixPup])
        maskField.sampling = zeroPad(maskField.sampling, (nPad - nPixPup) // 2)

        if has_opd:
            from mastsel.mavisPsf import defaultArrayBackend, defaultArrayCDtype, defaultArrayDtype

            maskOtf = Field(wvl, nPad, grid_diameter)
            coeff = defaultArrayBackend.asarray(2 * np.pi * 1e-9 / wvl, dtype=defaultArrayDtype)
            opd_map_backend = defaultArrayBackend.asarray(opd_map, dtype=defaultArrayDtype)
            phaseStat = coeff * opd_map_backend
            phaseStat = congrid(phaseStat, [nPixPup, nPixPup])
            phaseStat = zeroPad(phaseStat, (nPad - nPixPup) // 2)
            i_complex = defaultArrayCDtype(1j)
            maskOtf.sampling = maskField.sampling * np.exp(i_complex * phaseStat)
            maskOtf.pupilToOtf()
            maskOtf.sampling /= maskOtf.sampling.max()
            otf_tel = maskOtf.sampling
        else:
            otf_tel = None

        for psd_in in input_psds:
            psd = Field(wvl, int(N), freq_range, "rad")
            psd.sampling = psd_in / dk**2
            if nPad > N:
                psd.sampling = zeroPad(psd.sampling, (nPad - N) // 2)
                psd.width = psd.width * nPad / N

            # Old longExposurePsf behavior.
            from mastsel.mavisPsf import defaultArrayBackend, defaultArrayDtype

            dtype = (
                defaultArrayBackend.float32
                if defaultArrayDtype == defaultArrayBackend.float32
                else defaultArrayBackend.float64
            )
            pitch = 1.0 / psd.width
            if otf_tel is None:
                maskC = Field(maskField.wvl, maskField.N, pitch * maskField.N)
                maskC.sampling = np.copy(maskField.sampling)
                maskC.pupilToOtf()
                otf_tel_local = maskC.sampling
            else:
                otf_tel_local = otf_tel

            psd_pad = zeroPad(psd.sampling, psd.sampling.shape[0] // 2)
            coeff = defaultArrayBackend.asarray((psd.kk * psd.width) ** 2, dtype=dtype)
            B_phi = np.real(np.fft.ifft2(np.fft.ifftshift(psd_pad))) * coeff
            b0 = B_phi[0, 0]
            B_phi = np.fft.fftshift(B_phi)
            D_phi = 2.0 * b0 - (B_phi + B_phi.conj())
            otf_turb = np.exp(-0.5 * D_phi)
            otf_turb = congrid(otf_turb, [otf_turb.shape[0] // 2, otf_turb.shape[0] // 2])
            otf_system = otf_turb * otf_tel_local
            sampling = np.real(_ft_ft2_old(otf_system))

            nBig = int(np.ceil(sampling.shape[0] / ovrsmp / 2) * ovrsmp * 2)
            if nBig > sampling.shape[0]:
                sampling = zeroPad(sampling, (nBig - sampling.shape[0]) // 2)
            if ovrsmp > 1:
                nOvr = int(ovrsmp)
                nOut = int(sampling.shape[0] / nOvr)
                delta = (ovrsmp - 1) / 2
                if ovrsmp % 2:
                    sampling = np.roll(sampling, (int(delta), int(delta)), axis=(0, 1))
                else:
                    from scipy.ndimage import shift as a_shift

                    sampling = a_shift(sampling, (delta, delta), order=3, mode="constant")
                sampling = sampling.reshape((nOut, nOvr, nOut, nOvr)).mean(3).mean(1)

            if sampling.shape[0] > nPixPsf:
                start = (sampling.shape[0] - nPixPsf) // 2
                sampling = sampling[start : start + nPixPsf, start : start + nPixPsf]
            out.append(sampling)

    return out


@unittest.skipIf(not _load_fixture_paths(), "PSD fixtures not found. Run TIPTOP export script first.")
class TestPsfFixtureRegression(unittest.TestCase):
    def test_even_before_after_peak_within_1_percent(self):
        for fixture_path in _load_fixture_paths():
            data = np.load(fixture_path, allow_pickle=False)
            case_name = str(data["case_name"])
            N = int(data["N"])
            if N % 2 != 0:
                continue

            input_psds = np.asarray(data["input_psds"])
            mask = np.asarray(data["mask"])
            wavelength = np.asarray(data["wavelength"])
            nPixPup = int(data["nPixPup"])
            grid_diameter = float(data["grid_diameter"])
            freq_range = float(data["freq_range"])
            dk = float(data["dk"])
            nPixPsf = int(data["nPixPsf"])
            wvlRef = float(data["wvlRef"])
            oversampling = float(data["oversampling"])
            padPSD = bool(data["padPSD"])
            has_opd = bool(data["has_opd"])
            opd_map = np.asarray(data["opd_map"]) if has_opd else np.asarray([])

            old_set = _psdset_old_reference(
                input_psds,
                mask,
                wavelength,
                N,
                nPixPup,
                grid_diameter,
                freq_range,
                dk,
                nPixPsf,
                wvlRef,
                oversampling,
                opd_map,
                has_opd,
                padPSD,
            )
            new_set = psdSetToPsfSet(
                input_psds,
                mask,
                wavelength,
                N,
                nPixPup,
                grid_diameter,
                freq_range,
                dk,
                nPixPsf,
                wvlRef,
                oversampling,
                opdMap=opd_map if has_opd else None,
                padPSD=padPSD,
            )

            if isinstance(new_set[0], list):
                new_psfs = [item.sampling for row in new_set for item in row]
            else:
                new_psfs = [item.sampling for item in new_set]

            self.assertEqual(len(old_set), len(new_psfs), msg=f"{case_name}: mismatched PSF count")
            for idx, (old_psf, new_psf) in enumerate(zip(old_set, new_psfs)):
                old_aligned = _pad_or_crop_centered(old_psf, new_psf.shape[0])
                rel = _peak_rel_diff(old_aligned, new_psf)
                self.assertLessEqual(
                    rel,
                    PEAK_TOL,
                    msg=f"{case_name} psf#{idx}: peak rel diff={rel:.4f} > {PEAK_TOL:.2%}",
                )

    def test_even_vs_odd_peak_within_case_tolerance(self):
        for fixture_path in _load_fixture_paths():
            data = np.load(fixture_path, allow_pickle=False)
            case_name = str(data["case_name"])
            N = int(data["N"])
            if N % 2 != 0:
                continue

            input_psds = np.asarray(data["input_psds"])
            mask = np.asarray(data["mask"])
            wavelength = np.asarray(data["wavelength"])
            nPixPup = int(data["nPixPup"])
            grid_diameter = float(data["grid_diameter"])
            freq_range = float(data["freq_range"])
            dk = float(data["dk"])
            nPixPsf = int(data["nPixPsf"])
            wvlRef = float(data["wvlRef"])
            oversampling = float(data["oversampling"])
            padPSD = bool(data["padPSD"])
            has_opd = bool(data["has_opd"])
            opd_map = np.asarray(data["opd_map"]) if has_opd else None

            even_psf_set = psdSetToPsfSet(
                input_psds,
                mask,
                wavelength,
                N,
                nPixPup,
                grid_diameter,
                freq_range,
                dk,
                nPixPsf,
                wvlRef,
                oversampling,
                opdMap=opd_map,
                padPSD=padPSD,
            )
            if isinstance(even_psf_set[0], list):
                even_psfs = [item.sampling for row in even_psf_set for item in row]
            else:
                even_psfs = [item.sampling for item in even_psf_set]

            odd_N = N + 1
            odd_nPixPsf = nPixPsf + 1
            odd_nPixPup = nPixPup
            odd_grid = grid_diameter
            psd_step = freq_range / N
            odd_freq = odd_N * psd_step
            odd_psds = np.asarray([_expand_with_nyquist_row_col(psd) for psd in input_psds])
            odd_mask = mask
            odd_opd = opd_map

            odd_psf_set = psdSetToPsfSet(
                odd_psds,
                odd_mask,
                wavelength,
                odd_N,
                odd_nPixPup,
                odd_grid,
                odd_freq,
                dk,
                odd_nPixPsf,
                wvlRef,
                oversampling,
                opdMap=odd_opd,
                padPSD=padPSD,
            )
            if isinstance(odd_psf_set[0], list):
                odd_psfs = [item.sampling for row in odd_psf_set for item in row]
            else:
                odd_psfs = [item.sampling for item in odd_psf_set]

            self.assertEqual(len(even_psfs), len(odd_psfs), msg=f"{case_name}: even/odd PSF count mismatch")
            case_tol = _odd_even_peak_tol(case_name)
            for idx, (psf_e, psf_o) in enumerate(zip(even_psfs, odd_psfs)):
                rel = _peak_rel_diff(psf_e, psf_o)
                self.assertLessEqual(
                    rel,
                    case_tol,
                    msg=(
                        f"{case_name} psf#{idx}: even/odd peak rel diff={rel:.4f} "
                        f"> {case_tol:.2%} (case-specific tolerance)"
                    ),
                )
