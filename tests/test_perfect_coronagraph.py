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
        out = psdSetToPsfSet([np.zeros((n, n))], mask, WVL, nPixPup=npup, freq_range=3.0,
                             dk=1.0, nPixPsf=n, perfectCoronagraph=True)
        self.assertTrue(np.all(_host(out) == 0.0))

    @unittest.skipUnless(os.path.exists(FIXTURE), "SPHERE fixture not found.")
    def test_fixture_identity_and_scattered_fraction(self):
        d = np.load(FIXTURE, allow_pickle=False)
        psd = np.asarray(d["input_psds"])
        n, dk, fr = int(d["N"]), float(d["dk"]), float(d["freq_range"])
        wvl = float(d["wavelength"][0])
        b0 = _b0(psd[0], dk, fr, wvl, n)
        kw = dict(mask=np.asarray(d["mask"]), wavelength=d["wavelength"],
                  nPixPup=int(d["nPixPup"]), freq_range=fr, dk=dk)

        # native grid: coro = PSF - exp(-b0) PSF_DL
        psf = _host(psdSetToPsfSet(psd, nPixPsf=n, **kw))
        dl = _host(psdSetToPsfSet(np.zeros_like(psd), nPixPsf=n, **kw))
        coro = _host(psdSetToPsfSet(psd, nPixPsf=n, perfectCoronagraph=True, **kw))
        np.testing.assert_allclose(coro, psf - np.exp(-b0) * dl, rtol=0, atol=1e-12 * psf.max())
        self.assertAlmostEqual(coro.sum() / psf.sum(), 1 - np.exp(-b0), places=10)

        # decimated grid: same scattered fraction
        kw2 = dict(nPixPsf=n // 2, oversampling=2, **kw)
        psf2 = _host(psdSetToPsfSet(psd, **kw2))
        coro2 = _host(psdSetToPsfSet(psd, perfectCoronagraph=True, **kw2))
        self.assertAlmostEqual(coro2.sum() / psf2.sum(), 1 - np.exp(-b0), places=10)

    def test_monte_carlo_phase_screens(self):
        """Frame-by-frame perfect coronagraph on random screens vs the analytic model."""
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
        emp_var = phi.var()
        self.assertAlmostEqual(emp_var / b0, 1.0, delta=0.05)

        E = pup * np.exp(1j * phi)
        ft = lambda g: np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(g, axes=(-2, -1)), axes=(-2, -1)), axes=(-2, -1))
        I = np.abs(ft(E)) ** 2
        a = (pup * E).sum(axis=(1, 2)) / (pup**2).sum()
        dl = np.abs(ft(pup)) ** 2

        # radial annuli of 2 px; per-block estimates give the standard error by jackknife
        yy, xx = np.indices((n, n)) - n // 2
        ring = (np.hypot(xx, yy) // 2).astype(int).ravel()
        ring[ring >= n // 4] = n // 4
        prof = lambda im: np.bincount(ring, im.ravel(), n // 4 + 1) / np.bincount(ring, minlength=n // 4 + 1)

        def estimate(sel):
            coro = I[sel].mean(0) - np.abs(a[sel].mean()) ** 2 * dl
            return prof(coro) / I[sel].mean(0).sum()

        blocks = np.arange(nfr) // per_blk
        full = estimate(np.ones(nfr, bool))
        jk = np.array([estimate(blocks != b) for b in range(nblk)])
        se = np.sqrt((nblk - 1) / nblk * ((jk - jk.mean(0)) ** 2).sum(0))

        kw = dict(nPixPup=npup, freq_range=fr, dk=1.0, nPixPsf=n)
        model_psf = _host(psdSetToPsfSet([psd], mask, WVL, **kw))
        model = _host(psdSetToPsfSet([psd], mask, WVL, perfectCoronagraph=True, **kw))
        model = prof(model) / model_psf.sum()

        nsig = np.abs(model - full)[:-1] / se[:-1]            # drop the corner bin
        self.assertLess(nsig.max(), 4.0, msg=f"max deviation {nsig.max():.2f} sigma")
        halo_se = np.sqrt((nblk - 1) / nblk * ((jk.sum(1) - jk.sum(1).mean()) ** 2).sum())
        self.assertLess(abs(model.sum() - full.sum()), 4 * halo_se)
        print(f"\n[MC] var/b0={emp_var / b0:.4f}  max|dev|={nsig.max():.2f} sigma")

    def test_opd_map_not_supported(self):
        n, npup = 64, 24
        with self.assertRaises(ValueError):
            psdSetToPsfSet([np.zeros((n, n))], np.ones((npup, npup)), WVL, nPixPup=npup,
                           freq_range=3.0, dk=1.0, nPixPsf=n, perfectCoronagraph=True,
                           opdMap=np.zeros((npup, npup)))


if __name__ == "__main__":
    unittest.main()
