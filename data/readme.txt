The files in the folder are examples of data used or obtained in the jitter computation code. The parameters are the ones defined in params4jittercode.txt, unless otherwise specified. The files are the following:

- example_gaussian_psf.fits: Gaussian PSF for the 2nd star, normalized so that a PSF with SR of 1 would have had a flux of 1 e-, i. e. it takes into account the loss of flux in the PSF core due to SR < 1 and FWHM > diffraction.

- example_slope_noise.fits: Slope standard deviation (in mas) and bias factor in the case of the 2nd star sensed at 500 Hz.

- example_slope_noise_ngs1.fits: Same as example_slope_noise.fits for the 1st star sensed at 500 Hz.

- example_slope_noise_ngs3.fits: Same as example_slope_noise.fits for the 3rd star sensed at 500 Hz.

- windpsd_mavis.fits: PSD of tip and tilt (in nm2/Hz) due to wind/vibrations for MAVIS. Format: [nfreq,3]. 1st index: frequency in Hz. 2nd index: Tip PSD. 3rd index: Tilt PSD.

- windpsd_maory.fits: Same as windpsd_mavis.fits but for MAORY.

- minim_wind_noise12071nm2_psdmavis.fits: Table of residual variance on tip and tilt (in nm2) due to wind/vibrations for all the considered gains of the double integrator. Computed for the 2nd star sensed at 500 Hz. The noise level of 12071 nm2 corresponds to the slope error in example_slope_noise.fits. Format: [ngains,ngains,2].

- minim_wind_noise12071nm2_psdmaory.fits: same as minim_wind_noise12071nm2_psdmavis.fits but with the wind PSD from MAORY.

- psd_turb.fits: PSD of turbulent tip and tilt in nm2/Hz. Format: [nfreq,3]. 1st index: frequency in Hz. 2nd index: Tip PSD. 3rd index: Tilt PSD.

- res_noise_ngs.fits: Residual variance of tip and tilt (in nm2) for all gains explored in the noise computation. Computed for the 2nd star at 500 Hz.

- Cn.fits: Noise covariance matrix for all stars sensed at 500 Hz.

- Caa.fits: Turbulent tip/tilt covariance matrix for a single direction (in nm2).

- Cas.fits: Turbulent tip/tilt covariance matrix between a chosen direction of interest ([5,5] arcsec in cartesian) and the NGSs (in nm2).

- Css.fits: Turbulent tip/tilt covariance matrix between the different NGSs (in nm2).

- rec_tomo.fits: Tomographic reconstructor.

- cov_tomo_onaxis.fits: Resulting covariance matrix of tip/tilt on axis for the tomography contribution.

- cov_noise_onaxis.fits: Resulting covariance matrix of tip/tilt on axis for the noise contribution. Computed using Cn.fits.

- jitter_map.fits: Resulting jitter map for the configuration described in params4jittercode.txt (frequency imposed at 500 Hz). Format: [ndirections,5]. 1st & 2nd indexes: X, Y coordinates in arcsec. 3rd & 4th indexes: tip/tilt standard deviation along the major & minor axes (in mas). 5th index: Angle between the major axis and the X axis in degrees (counterclockwise).
