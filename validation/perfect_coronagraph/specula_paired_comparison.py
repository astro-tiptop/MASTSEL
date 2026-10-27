"""
Perfect coronagraph: frame-by-frame reference vs long-exposure formulas, on the
residual fields of a SPECULA end-to-end run (validation only, not a test).

Inputs : the residual electric field saved by SPECULA (DataStore of
         'prop.out_source_ngs_ef', here every 4 ms), see coro_override.yml.
Outputs: ratio tables printed per band and annulus (lambda/D), and
         specula_eris_reference.npz with radial profiles and 128x128 crops of
         the long-exposure PSF and of the frame-by-frame coronagraphic image.
Run    : python specula_paired_comparison.py <ef.fits> <t_end [s]> <out.npz>

Frame-by-frame reference: in each frame the projection a P of the field on the
pupil mode is removed (a = pupil average of E), as SPECULA PsfCoronagraph.
Compared with (a) <I> - |<a>|^2 PSF_DL (long-exposure coherent removal),
(b) <I> - SR PSF_DL and (c) the MASTSEL formula, i.e. the exact Gaussian average
<I> - 2 Re[C conj(FT P)] + SR |FT P|^2, C = FT[P (P conv g)] / S, with the
empirical g = OTF_LE / OTF_tel.
"""
import sys
import numpy as np
import cupy as cp
from astropy.io import fits

ef_file, t_end, out_file = sys.argv[1], float(sys.argv[2]), sys.argv[3]
dt, t0, nd = 0.004, 0.1, 4                       # saved-field period, start time, px per lambda/D
bands = (('J', 1250), ('H', 1650), ('K', 2200), ('L', 3800))
edges = np.array([0, 0.5, 1, 1.5, 2, 3, 5, 10, 20, 40, 80])

ef = fits.getdata(ef_file)                        # (frames, 2, n, n): amplitude, phase [nm]
nf = min(ef.shape[0], int(round((t_end - t0) / dt)))
P = (cp.asarray(ef[0, 0]) > 0).astype(cp.float64)
npup = P.shape[0]; n = nd * npup; S = float(P.sum())
Pp = cp.zeros((n, n)); Pp[:npup, :npup] = P
FP = cp.fft.fft2(Pp); dl = cp.abs(FP)**2; otf_tel = cp.fft.ifft2(dl)
yy, xx = np.mgrid[:n, :n]; r = np.hypot(xx - n // 2, yy - n // 2) / nd
c, h = n // 2, 64
out = {'edges': edges, 'frames': nf, 't_end': t0 + dt * nf}
print(f'frames used: {nf} (t = {t0} .. {t0 + dt * nf:.2f} s)')
for band, wl in bands:
    I = cp.zeros((n, n)); Ipf = cp.zeros((n, n)); asum = 0j
    for k in range(nf):
        E = cp.zeros((n, n), cp.complex128)
        ph = cp.asarray(ef[k, 1], dtype=cp.float64) * (2 * np.pi / wl)
        E[:npup, :npup] = P * cp.exp(1j * (ph - ph[P > 0].mean()))
        a = E.sum() / S; asum += complex(a)
        F = cp.fft.fft2(E); I += cp.abs(F)**2; Ipf += cp.abs(F - a * FP)**2
    I /= nf; Ipf /= nf; am = asum / nf
    otf_le = cp.fft.ifft2(I)
    g = cp.where(cp.abs(otf_tel) > 1e-6 * float(cp.abs(otf_tel).max()), otf_le / otf_tel, 0)
    C = cp.fft.fft2(Pp * cp.fft.ifft2(FP * cp.fft.fft2(g))) / S
    sr = float(cp.real(otf_le).sum() / S**2)                          # <|a|^2>
    ims = {'fbf': Ipf, 'le_coherent': I - abs(am)**2 * dl, 'sr_removal': I - sr * dl,
           'model': I - 2 * cp.real(C * cp.conj(FP)) + sr * dl, 'psf': I}
    pk = float(I.max())
    ims = {k: cp.asnumpy(cp.fft.fftshift(v)) / pk for k, v in ims.items()}   # contrast units
    prof = {k: np.array([np.median(v[(r >= lo) & (r < hi)]) for lo, hi in zip(edges[:-1], edges[1:])])
            for k, v in ims.items()}
    out.update({f'{band}_{k}_profile': v for k, v in prof.items()})
    out[f'{band}_sr'] = sr
    out[f'{band}_psf_crop'] = ims['psf'][c - h:c + h, c - h:c + h].astype(np.float32)
    out[f'{band}_coro_crop'] = ims['fbf'][c - h:c + h, c - h:c + h].astype(np.float32)
    print(f"\n== {band}: SR {sr:.3f}  |<a>|^2 {abs(am)**2:.3f}   ratio to frame-by-frame")
    print(f"{'r':>9s} {'fbf':>9s} {'LE coh':>7s} {'SR rem':>7s} {'model':>7s}")
    for i, (lo, hi) in enumerate(zip(edges[:-1], edges[1:])):
        f = prof['fbf'][i]
        print(f"{lo:4.1f}-{hi:<4.0f} {f:9.2e} {prof['le_coherent'][i]/f:7.3f} {prof['sr_removal'][i]/f:7.3f} {prof['model'][i]/f:7.3f}")
np.savez_compressed(out_file, **out)
