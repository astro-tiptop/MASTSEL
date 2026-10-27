#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import unittest

import numpy as np

from mastsel.mavisLO import cpuArray
from mastsel.mavisPsf import defaultArrayBackend, pupilOnGrid, psdSetToPsfSet

FIXTURE = os.path.join(os.path.dirname(__file__), "fixtures", "psd", "SPHERE.npz")
WVL = 1.65e-6


def _host(psf_set):
    return np.asarray(cpuArray(psf_set[0].sampling), dtype=np.float64)


def _b0(psd, dk, freq_range, wvl, n):
    """Phase variance (rad^2) as computed in longExposurePsf (even N, OTF on the PSD grid)."""
    return float(np.sum(psd / dk**2) / n**2 * (2 * np.pi / wvl * freq_range) ** 2 / 4)


class TestPerfectCoronagraph(unittest.TestCase):
    def test_zero_psd_gives_zero_image(self):
        n, npup = 64, 24
        mask = np.ones((npup, npup))
        kw = dict(nPixPup=npup, freq_range=3.0, dk=1.0, nPixPsf=n)
        out = _host(psdSetToPsfSet([np.zeros((n, n))], mask, WVL, perfectCoronagraph=True, **kw))
        dl = _host(psdSetToPsfSet([np.zeros((n, n))], mask, WVL, **kw))
        # not exactly 0: rounding of the three FFT terms
        self.assertLess(np.abs(out).max(), 1e-12 * dl.max())

    @unittest.skipUnless(os.path.exists(FIXTURE), "SPHERE fixture not found.")
    def test_fixture_scattered_fraction_and_positivity(self):
        d = np.load(FIXTURE, allow_pickle=False)
        psd = np.asarray(d["input_psds"])
        n = int(d["N"])
        kw = dict(mask=np.asarray(d["mask"]), wavelength=d["wavelength"], nPixPup=int(d["nPixPup"]),
                  freq_range=float(d["freq_range"]), dk=float(d["dk"]), nPixPsf=n)
        psf = _host(psdSetToPsfSet(psd, **kw))
        dl = _host(psdSetToPsfSet(np.zeros_like(psd), **kw))
        coro = _host(psdSetToPsfSet(psd, perfectCoronagraph=True, **kw))
        # 1 - SR, with SR = peak ratio for a binary pupil (grey edges: ~4e-5 difference)
        self.assertAlmostEqual(coro.sum() / psf.sum(), 1 - psf.max() / dl.max(), delta=1e-3)
        self.assertGreater(coro.min(), -1e-9 * psf.max())

    def test_monte_carlo_frame_by_frame(self):
        """<|FT(E - aP)|^2> on random Gaussian screens vs the analytic perfect coronagraph."""
        n, npup, fr, nblk, per_blk = 64, 24, 3.0, 20, 13   # pitch 1/fr, pupil = 24 px
        rng = np.random.default_rng(1234)
        mask = np.ones((npup, npup))
        pup = np.asarray(cpuArray(pupilOnGrid(mask, npup, n, defaultArrayBackend)), dtype=np.float64)

        # isotropic power law with a low-frequency cut, zero near Nyquist (symmetric spectrum)
        f = np.fft.fftshift(np.fft.fftfreq(n, 1.0 / n))
        kr = np.hypot(*np.meshgrid(f, f))
        psd = np.where((kr >= 1.5) & (kr < n / 2 - 1), np.maximum(kr, 1e-9) ** (-11 / 3), 0.0)
        coeff = (2 * np.pi / WVL * fr) ** 2 / 4
        psd *= 0.3 / (np.sum(psd) / n**2 * coeff)           # b0 = 0.3 rad^2
        b0 = _b0(psd, 1.0, fr, WVL, n)

        # Re(FFT(sqrt(S) w)) * sqrt(2)/N has covariance B = IFFT(S); Re and Im are independent screens
        spec = np.sqrt(np.fft.ifftshift(psd) * coeff)
        nfr = nblk * per_blk
        w = (rng.standard_normal((nfr // 2 + 1, n, n)) + 1j * rng.standard_normal((nfr // 2 + 1, n, n))) / np.sqrt(2)
        x = np.fft.fft2(spec * w) * np.sqrt(2) / n
        phi = np.concatenate([x.real, x.imag])[:nfr]
        self.assertAlmostEqual(phi.var() / b0, 1.0, delta=0.05)

        E = pup * np.exp(1j * phi)
        ft = lambda g: np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(g, axes=(-2, -1)), axes=(-2, -1)), axes=(-2, -1))
        a = (pup * E).sum(axis=(1, 2)) / (pup**2).sum()
        I = np.abs(ft(E)) ** 2
        Ic = np.abs(ft(E - a[:, None, None] * pup)) ** 2
        dl = np.abs(ft(pup)) ** 2

        # radial annuli of 2 px; blocks of frames give the standard error by jackknife
        yy, xx = np.indices((n, n)) - n // 2
        ring = (np.hypot(xx, yy) // 2).astype(int).ravel()
        ring[ring >= n // 4] = n // 4
        prof = lambda im: np.bincount(ring, im.ravel(), n // 4 + 1) / np.bincount(ring, minlength=n // 4 + 1)
        blocks = np.arange(nfr) // per_blk

        def jackknife(stat):
            full = stat(np.ones(nfr, bool))
            jk = np.array([stat(blocks != b) for b in range(nblk)])
            return full, np.sqrt((nblk - 1) / nblk * ((jk - jk.mean(0)) ** 2).sum(0))

        def frame_by_frame(sel):
            return np.append(prof(Ic[sel].mean(0)) / I[sel].mean(0).sum(), (np.abs(a[sel]) ** 2).mean())

        def old_formula(sel):
            return prof(I[sel].mean(0) - np.abs(a[sel].mean()) ** 2 * dl) / I[sel].mean(0).sum()

        mc, se = jackknife(frame_by_frame)
        old, old_se = jackknife(old_formula)

        kw = dict(nPixPup=npup, freq_range=fr, dk=1.0, nPixPsf=n)
        psf = _host(psdSetToPsfSet([psd], mask, WVL, **kw))
        coro = _host(psdSetToPsfSet([psd], mask, WVL, perfectCoronagraph=True, **kw))
        model = prof(coro) / psf.sum()

        nsig = np.abs(model - mc[:-1])[:-1] / se[:-2]       # drop the corner bin and <|a|^2>
        self.assertLess(nsig.max(), 4.0, msg=f"max deviation {nsig.max():.2f} sigma")
        centre = (coro[n // 2, n // 2] / psf.sum() - Ic[:, n // 2, n // 2].mean() / I.mean(0).sum())
        a2_model = 1 - coro.sum() / psf.sum()
        self.assertLess(abs(a2_model - mc[-1]), 4 * se[-1])
        old_nsig = np.abs(old - mc[:-1])[0] / np.hypot(se[0], old_se[0])
        self.assertGreater(old_nsig, 4.0)   # the old long-exposure formula fails at the centre
        print(f"\n[MC] max|dev|={nsig.max():.2f} sigma, per annulus={np.round(nsig, 2)}, "
              f"centre diff={centre:.3e}, <|a|^2> MC={mc[-1]:.4f}+-{se[-1]:.4f} lib={a2_model:.4f}, "
              f"old formula vs frame-by-frame, annulus 0: {old_nsig:.1f} sigma")

    def test_opd_map_not_supported(self):
        n, npup = 64, 24
        with self.assertRaises(ValueError):
            psdSetToPsfSet([np.zeros((n, n))], np.ones((npup, npup)), WVL, nPixPup=npup,
                           freq_range=3.0, dk=1.0, nPixPsf=n, perfectCoronagraph=True,
                           opdMap=np.zeros((npup, npup)))


if __name__ == "__main__":
    unittest.main()
