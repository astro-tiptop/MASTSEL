# MASTSEL

This library was developed with 2 goals: support the Asterism Selection for
MAVIS instrument (https://mavis-ao.org/) and managing the computation of the
jitter (i.e. tip/tilt) error in Adaptive Optics simulations done in the Fourier
domain. It is used by TIPTOP (https://github.com/astro-tiptop/TIPTOP).

The main features are located in:

- `mastsel/mavisLO.py` class that computes the jitter ellipses that can be
- convolved with High Orders (HO, i.e. aberrations of higher spatial
- frequencies than tip/tilt) Point Spread Functions (PSF) to get the PSFs that
- consider both the effect of HO and Low Orders (LO, i.e. tip/tilt).

- `mastsel/mavisPsf.py` that contains a set of methods and classes to compute
- short and long exposure PSF from Power Spectral Densities (PSD), Strehl
- Ratios, radial profiles, encircled energies (and other quantities) from PSF,
- to convolve kernels with PSFs, …

Reference: section 5 “LOW ORDER PART OF THE PSF” of Benoit et al. "TIPTOP:
a new tool to efficiently predict your favorite AO PSF" SPIE 2020 (ARXIV:
https://doi.org/10.48550/arXiv.2101.06486).
