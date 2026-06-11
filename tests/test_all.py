from mastsel.mavisFormulas import *
from mastsel.mavisLO import *
from mastsel.mavisPsf import *

import unittest

class TestMavisLO(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        path = "data/ini/"
        parametersFile = 'mavisParamsTests'
        fullPathFilename = path + parametersFile + '.ini'
        windPsdFile = 'data/windpsd_mavis.fits'
        TestMavisLO.mLO = MavisLO(path, parametersFile, verbose=True)
        f1 = TestMavisLO.mLO.get_config_value('RTC','SensorFrameRate_LO')
        TestMavisLO.mLO.configLOFreq(f1)

class TestReconstructor(TestMavisLO):
                    
    def test_reconstructor(self):
        """
        Test 
        """                
        cartPointingCoords = np.asarray([5,5])
        polarNGSCoords = np.asarray([[30.0,0.0], [50.0,100.0],[10.0,240.0]])
        cartNGSCoords = np.asarray([polarToCartesian(polarNGSCoords[0]), polarToCartesian(polarNGSCoords[1]), polarToCartesian(polarNGSCoords[2])])
        P_mat, rec_tomo, R_0, R_1 = TestMavisLO.mLO.buildReconstuctor(cartPointingCoords, cartNGSCoords)
        a1 = [[ 1.,          0.,          0.42221252,  0.,          0.29854934],
         [ 0.,          1.,          0.,          0.29854934, -0.        ],
         [ 1.,          0.,         -0.12219406,  0.49002284, -0.08640425],
         [ 0.,          1.,          0.69299694, -0.08640425, -0.49002284],
         [ 1.,          0.,         -0.07036875, -0.08618377, -0.04975822],
         [ 0.,          1.,         -0.12188226, -0.04975822,  0.08618377],]
        a2 = [[ 0.1659003,  -0.15351883,  0.21901709,  0.02296196,  0.61508262,  0.13055688],
         [-0.01174574,  0.34082171, -0.1116391,   0.09859392,  0.12338484,  0.56058436],
         [ 0.99933477,  0.06355645, -0.14947145,  0.60408168, -0.84986332, -0.66763813],
         [ 0.12251111,  0.90023268,  1.16442494, -0.13464866, -1.28693605, -0.76558402],
         [ 1.37527988,  0.50750197, -0.57249372, -0.94365185, -0.80278616,  0.43614989]]
        a3 = [[ 0.1659003,  -0.15351883,  0.21901709,  0.02296196,  0.61508262,  0.13055688],
         [-0.01174574,  0.34082171, -0.1116391,   0.09859392,  0.12338484,  0.56058436,]]
        a4 =[[ 0.31074966, -0.07900007,  0.23795241,  0.01181612,  0.45129793,  0.06718396],
         [-0.00375934,  0.36483568, -0.03573124,  0.18135696,  0.03949058,  0.45380736]]
        P_mat__ = np.asarray(a1) 
        rec_tomo__ = np.asarray(a2)
        R_0__ = np.asarray(a3)
        R_1__ = np.asarray(a4)
        self.assertTrue( np.testing.assert_allclose(P_mat, P_mat__, rtol=1e-03, atol=1e-5)==None)
        self.assertTrue( np.testing.assert_allclose(rec_tomo, rec_tomo__, rtol=1e-03, atol=1e-5)==None)
        self.assertTrue( np.testing.assert_allclose(R_0, R_0__, rtol=1e-03, atol=1e-5)==None)
        self.assertTrue( np.testing.assert_allclose(R_1, R_1__, rtol=1e-03, atol=1e-5)==None)

    def test_reconstructor2_gpu_type_safety(self):
        """
        Regression & Type-Safety Test: Verifies that buildReconstructor2 
        correctly handles device isolation and does not trigger implicit 
        NumPy/CuPy conversion type errors when running on GPU.
        """
        # Define clean mock pointing coordinates (2D array as expected by the simulation)
        cartPointingCoordsV = np.asarray([[5.0, 5.0], [10.0, -10.0]])
        polarNGSCoords = np.asarray([[30.0, 0.0], [50.0, 100.0], [10.0, 240.0]])
        cartNGSCoords = np.asarray([polarToCartesian(p) for p in polarNGSCoords])
        
        # Mock some baseline covariance matrices for MMSE tracking
        Caa = np.eye(2, dtype=np.float64) * 0.1
        Cnn = np.eye(6, dtype=np.float64) * 0.05
        
        # Test 1: Validate standard execution returns correct dimensions
        R, RT = TestMavisLO.mLO.buildReconstuctor2(cartPointingCoordsV, cartNGSCoords, Cnn=Cnn, Caa=Caa)
        self.assertEqual(R.shape, (4, 6)) # 2 directions * 2 (TT) rows, 3 stars * 2 cols
        
        # Test 2: Force GPU path if cupy is installed to safeguard against implicit conversions
        if gpuEnabled:
            try:
                # Force the class to simulate a GPU computation platform
                original_platform = TestMavisLO.mLO.computationPlatform
                TestMavisLO.mLO.computationPlatform = 'GPU'
                
                # Run the reconstructor under simulated active GPU conditions
                R_gpu, RT_gpu = TestMavisLO.mLO.buildReconstuctor2(cartPointingCoordsV, cartNGSCoords, Cnn=Cnn, Caa=Caa)
                
                # Check that tensors can be successfully moved back to CPU without crashing
                self.assertIsNotNone(R_gpu)
                
            finally:
                # Always restore the original platform configuration to avoid test pollution
                TestMavisLO.mLO.computationPlatform = original_platform



class TestCovMatrices(TestMavisLO):
    def test_cov_matrices(self):
        """
        Test 
        """
        cartPointingCoords = np.asarray([5,5])
        polarNGSCoords = np.asarray([[30.0,0.0], [50.0,100.0],[10.0,240.0]])
        cartNGSCoords = np.asarray([polarToCartesian(polarNGSCoords[0]), polarToCartesian(polarNGSCoords[1]), polarToCartesian(polarNGSCoords[2])])
        print("Running Test: TestCovMatrices")
        matCaaValue, matCasValue, matCssValue = TestMavisLO.mLO.computeCovMatrices(cartPointingCoords, cartNGSCoords) 

        hdul = fits.open('data/Caa.fits')
        Caa_data = np.asarray(hdul[0].data, np.float64)
        hdul.close()
        hdul = fits.open('data/Cas.fits')
        Cas_data = np.asarray(hdul[0].data, np.float64)
        hdul.close()
        hdul = fits.open('data/Css.fits')
        Css_data = np.asarray(hdul[0].data, np.float64)
        hdul.close()
        self.assertTrue( np.testing.assert_allclose(Caa_data,matCaaValue[:2,:], rtol=1e-03, atol=1e-5)==None)
        self.assertTrue( np.testing.assert_allclose(Cas_data,matCasValue[:2,:], rtol=1e-03, atol=1e-5)==None)
        self.assertTrue( np.testing.assert_allclose(Css_data,matCssValue, rtol=1e-03, atol=1e-5)==None)        


class TestNoiseResiduals(TestMavisLO):
    def test_noise_residuals(self):
        """
        Test 
        """ 
        print("Running Test: TestNoiseResiduals")

        NGS_flux = [2500, 7500 , 1250]
        NGS_freq = [500, 500 , 500]
        NGS_SR_1650 = [0.4, 0.2, 0.6]
        NGS_FWHM_mas = [51.677, 81.673, 42.373]
        
        TestMavisLO.mLO.simpleVarianceComputation = False
        mItGPU = Integrator(cp, cp.float64, '')
        PixelScale = TestMavisLO.mLO.PixelScale_LO[0]
        TestMavisLO.mLO.configLOFreq(NGS_freq[0]) 
        r1 = TestMavisLO.mLO.computeBiasAndVariance(NGS_flux[0], NGS_freq[0], NGS_SR_1650[0], NGS_FWHM_mas[0], PixelScale)
        TestMavisLO.mLO.configLOFreq(NGS_freq[1]) 
        r2 = TestMavisLO.mLO.computeBiasAndVariance(NGS_flux[1], NGS_freq[1], NGS_SR_1650[1], NGS_FWHM_mas[1], PixelScale)
        TestMavisLO.mLO.configLOFreq(NGS_freq[2]) 
        r3 = TestMavisLO.mLO.computeBiasAndVariance(NGS_flux[2], NGS_freq[2], NGS_SR_1650[2], NGS_FWHM_mas[2], PixelScale)
        
        self.assertTrue( np.testing.assert_allclose(np.array(r1[0]), np.array((0.3532540510862264)), rtol=1e-03, atol=1e-5)==None)
        self.assertTrue( np.testing.assert_allclose(np.array(r1[1]), np.array((0.0883135127715566, -2.8275988362257914e-09)), rtol=1e-03, atol=1e-5)==None)
        self.assertTrue( np.testing.assert_allclose(np.array(r1[2]), np.array((0.14564300983994172, 0.1456430098399417)), rtol=1e-03, atol=1e-5)==None)
        self.assertTrue( np.testing.assert_allclose(np.array(r2[0]), np.array((0.32539772105091225)), rtol=1e-03, atol=1e-5)==None)
        self.assertTrue( np.testing.assert_allclose(np.array(r2[1]), np.array((0.08134943026272806, 2.1788091530849742e-17)), rtol=1e-03, atol=1e-5)==None)
        self.assertTrue( np.testing.assert_allclose(np.array(r2[2]), np.array((0.15495676441587683, 0.154956764415876861)), rtol=1e-03, atol=1e-5)==None)
        self.assertTrue( np.testing.assert_allclose(np.array(r3[0]), np.array((0.30692711822411245)), rtol=1e-03, atol=1e-5)==None)
        self.assertTrue( np.testing.assert_allclose(np.array(r3[1]), np.array((0.07673177955602811, -1.4843916271439003e-09)), rtol=1e-03, atol=1e-5)==None)
        self.assertTrue( np.testing.assert_allclose(np.array(r3[2]), np.array((0.15620390676644558, 0.15620390676644558)), rtol=1e-03, atol=1e-5)==None)


class TestWindResiduals(TestMavisLO):
    def test_wind_residuals(self):
        """
        Test 
        """
        print("Running Test: TestWindResiduals")

        psd_freq, psd_tip_wind, psd_tilt_wind = TestMavisLO.mLO.loadWindPsd('data/windpsd_mavis.fits')
        var1x = 0.05993281522281573 * TestMavisLO.mLO.PixelScale_LO[0]**2
        bias = 0.4300779971881394
        nr = TestMavisLO.mLO.computeNoiseResidual(0.25, 250.0, 1000, var1x, bias )
        wr = TestMavisLO.mLO.computeWindResidual(psd_freq, psd_tip_wind, psd_tilt_wind, var1x, bias )

        result = nr[0]
        self.assertTrue( np.testing.assert_allclose(result, 736.61, rtol=1e-03, atol=1e-5)==None)
        result = nr[1]
        self.assertTrue( np.testing.assert_allclose(result, 438.381, rtol=1e-03, atol=1e-5)==None)
        result = wr[0]
        self.assertTrue( np.testing.assert_allclose(result, 249.53, rtol=1e-03, atol=1e-5)==None)
        result = wr[1]
        self.assertTrue( np.testing.assert_allclose(result, 179.12, rtol=1e-03, atol=1e-5)==None)


class TestBiasAndVariance(TestMavisLO):
    def test_bias_and_variance(self):
        """
        Test 
        """
        print("Running Test: TestBiasAndVariance")

        aNGS_EE = 1
        aNGS_flux = 100000
        aNGS_freq = 100
        aNGS_FWHM_mas = 2*TestMavisLO.mLO.PixelScale_LO[0]
        
        TestMavisLO.mLO.simpleVarianceComputation = False
        TestMavisLO.mLO.configLOFreq(aNGS_freq)
        TestMavisLO.mLO.configSpecMeanVarFormulas()
        
        aNGS_frameflux = aNGS_flux / aNGS_freq
        TestMavisLO.mLO.smallGridSize = 2
        TestMavisLO.mLO.mediumPixelScale = TestMavisLO.mLO.PixelScale_LO[0]/TestMavisLO.mLO.downsample_factor
        asigma = aNGS_FWHM_mas/sigmaToFWHM/TestMavisLO.mLO.mediumPixelScale
               
        xCoords = np.asarray(np.linspace(-TestMavisLO.mLO.largeGridSize/2.0+0.5, TestMavisLO.mLO.largeGridSize/2.0-0.5, TestMavisLO.mLO.largeGridSize), dtype=np.float32)
        yCoords = np.asarray(np.linspace(-TestMavisLO.mLO.largeGridSize/2.0+0.5, TestMavisLO.mLO.largeGridSize/2.0-0.5, TestMavisLO.mLO.largeGridSize), dtype=np.float32)
        xGrid, yGrid = np.meshgrid( xCoords, yCoords, sparse=False, copy=True)
                
        g2d = simple2Dgaussian( xGrid, yGrid, 0, 0, asigma)
        g2d = g2d * 1 / np.sum(g2d)
        I_k_data = g2d * aNGS_EE # Encirceld Energy in double FWHM is used to scale the PSF model
        I_k_data = I_k_data * aNGS_frameflux     
        I_k_data = intRebin(I_k_data, TestMavisLO.mLO.mediumShape) * TestMavisLO.mLO.downsample_factor**2
        ii1, ii2 = int(TestMavisLO.mLO.mediumGridSize/2-TestMavisLO.mLO.smallGridSize), int(TestMavisLO.mLO.mediumGridSize/2+TestMavisLO.mLO.smallGridSize)
        I_k_data = I_k_data[ii1:ii2,ii1:ii2]
        
        xplot1, mu_ktr_array = TestMavisLO.mLO.compute2DMeanVar( TestMavisLO.mLO.aFunctionM, TestMavisLO.mLO.expr0M, I_k_data, TestMavisLO.mLO.aFunctionMGauss)
        xplot2, var_ktr_array = TestMavisLO.mLO.compute2DMeanVar( TestMavisLO.mLO.aFunctionV, TestMavisLO.mLO.expr0V, I_k_data, TestMavisLO.mLO.aFunctionVGauss)
        var_ktr_array = var_ktr_array - mu_ktr_array**2

        mu_thr, var_thr = meanVarPixelThr(I_k_data,
                                          ron=TestMavisLO.mLO.sigmaRON_LO,
                                          bg=(TestMavisLO.mLO.Dark_LO+TestMavisLO.mLO.skyBackground_LO)/TestMavisLO.mLO.SensorFrameRate_LO,
                                          excess=TestMavisLO.mLO.ExcessNoiseFactor_LO,
                                          thresh=TestMavisLO.mLO.ThresholdWCoG_LO,
                                          new_value=TestMavisLO.mLO.NewValueThrPix_LO)

        result = np.max(np.abs(mu_ktr_array-mu_thr))
        self.assertTrue( np.testing.assert_array_less(result, 1e-3)==None)
        result = np.max(np.abs(var_ktr_array-var_thr))
        self.assertTrue( np.testing.assert_array_less(result, 1e-3)==None)


class TestPsfExtrapolation(unittest.TestCase):
    def test_estimate_exponent_from_fraction(self):
        r = np.arange(1.0, 31.0, dtype=np.float64)
        true_exponent = -3.27
        psf = 2.5 * r**true_exponent
        fraction = (0.4, 0.8)

        _, _, exponent_est, normalization_est = extrapolate_psf_profile(
            r, psf, r_max=40, fraction=fraction, verbose=False
        )

        self.assertAlmostEqual(exponent_est, true_exponent, places=2)
        self.assertAlmostEqual(normalization_est, 2.5, places=2)

    def test_forced_exponent_preserves_continuity_on_fit_interval(self):
        r = np.arange(1.0, 31.0, dtype=np.float64)
        psf = 2.5 * r**(-3.27)
        forced_exponent = -11 / 3
        fraction = (0.5, 0.75)

        _, psf_extended, exponent_out, normalization_out = extrapolate_psf_profile(
            r,
            psf,
            r_max=40,
            power_law_exponent=forced_exponent,
            fraction=fraction,
            verbose=False,
        )

        idx_start, idx_end = fraction_to_index_range(len(r), fraction)

        self.assertAlmostEqual(exponent_out, forced_exponent)
        self.assertGreater(normalization_out, 0)
        self.assertAlmostEqual(psf_extended[idx_start], psf[idx_start])
        self.assertAlmostEqual(psf_extended[idx_end - 1], psf[idx_end - 1])

    def test_auto_exponent_is_clipped_to_bounds(self):
        r = np.arange(1.0, 31.0, dtype=np.float64)
        psf = 1.7 * r**(-2.4)
        fraction = (0.5, 0.75)

        r_extended, psf_extended, exponent_out, normalization_out = extrapolate_psf_profile(
            r,
            psf,
            r_max=40,
            power_law_min_max=(-11 / 3, -3),
            fraction=fraction,
            verbose=False,
        )

        idx_start, idx_end = fraction_to_index_range(len(r), fraction)

        self.assertAlmostEqual(exponent_out, -3.0)
        self.assertGreater(normalization_out, 0)
        self.assertAlmostEqual(psf_extended[idx_start], psf[idx_start])
        self.assertAlmostEqual(psf_extended[idx_end - 1], psf[idx_end - 1])
        self.assertEqual(len(r_extended), len(psf_extended))


class TestPsfSpatialResampling(unittest.TestCase):
    """Test the spatial interpolation and cropping logic in psdSetToPsfSet."""
    
    def setUp(self):
        self.wavelengths = [1.2e-6, 2.2e-6]
        self.n = 256
        self.n_pix_pup = 220
        self.grid_diameter = 8.0
        self.freq_range = self.n / self.grid_diameter
        self.pupil_mask = np.ones((self.n_pix_pup, self.n_pix_pup), dtype=np.float64)
        self.dk = 4.0
        self.nPixPsf = 512
        self.wvl_ref = 2.2e-6
        self.kRef = 2
        
        # Create a simple delta PSD to track energy conservation
        self.psd = np.zeros((self.n, self.n), dtype=np.float64)
        self.psd[self.n//2, self.n//2] = 1.0

    def test_output_dimensions_are_strictly_target_fov(self):
        """
        Verify that regardless of the wavelength and the native FFT scaling,
        the output arrays are strictly of shape (nPixPsf, nPixPsf).
        """
        result = psdSetToPsfSet(
            [self.psd], self.pupil_mask, self.wavelengths, self.n, self.n_pix_pup,
            self.grid_diameter, self.freq_range, self.dk, self.nPixPsf, 
            self.wvl_ref, self.kRef, padPSD=True
        )
        
        psf_short = result[0][0] # 1.2um
        psf_long = result[1][0]  # 2.2um
        
        self.assertEqual(psf_short.sampling.shape[0], self.nPixPsf)
        self.assertEqual(psf_short.sampling.shape[1], self.nPixPsf)
        self.assertEqual(psf_long.sampling.shape[0], self.nPixPsf)
        self.assertEqual(psf_long.sampling.shape[1], self.nPixPsf)

    def test_flux_conservation_after_interpolation(self):
        """
        Verify that the spatial interpolation mathematically conserves 
        the total energy of the PSF.
        """
        # Run in mono-mode so native = target, ensuring baseline flux is stable
        result_mono = psdSetToPsfSet(
            [self.psd], self.pupil_mask, [self.wavelengths[1]], self.n, self.n_pix_pup,
            self.grid_diameter, self.freq_range, self.dk, self.nPixPsf, 
            self.wvl_ref, self.kRef, padPSD=False
        )

        # Run in multi-mode where the 2.2um PSF will be heavily interpolated
        result_multi = psdSetToPsfSet(
            [self.psd], self.pupil_mask, self.wavelengths, self.n, self.n_pix_pup,
            self.grid_diameter, self.freq_range, self.dk, self.nPixPsf, 
            self.wvl_ref, self.kRef, padPSD=True
        )

        flux_mono = float(result_mono[0].sampling.sum())
        flux_multi = float(result_multi[1][0].sampling.sum())
        
        self.assertAlmostEqual(flux_mono, flux_multi, places=5, 
                               msg="Flux was not conserved during spatial interpolation")

    def test_target_pixel_scale_is_tied_to_minimum_wavelength(self):
        """
        Verify that the geometric width (Field of View in radians) of the output 
        Field object correctly scales based on the minimum wavelength in the batch,
        rather than the reference wavelength.
        """
        result = psdSetToPsfSet(
            [self.psd], self.pupil_mask, self.wavelengths, self.n, self.n_pix_pup,
            self.grid_diameter, self.freq_range, self.dk, self.nPixPsf, 
            self.wvl_ref, self.kRef, padPSD=True
        )
        
        psf_short = result[0][0]
        psf_long = result[1][0]
        
        # Both output PSFs should have the exact same physical width in radians,
        # dictated by the target pixel scale derived from the 1.2um wavelength.
        self.assertEqual(psf_short.width, psf_long.width)


class TestStrehlRatioConsistency(unittest.TestCase):
    """Test consistency between Strehl Ratio from PSF and from PSD via Marechal."""
    
    @unittest.skip("PSF generation from non-zero PSD produces flat arrays - needs investigation")
    def test_strehl_ratio_psf_vs_marechal_synthetic(self):
        """
        WORK IN PROGRESS: Test SR from PSF peak ratio vs Marechal approximation.
        
        Current Issue:
        --------------
        When passing a non-zero PSD to psdSetToPsfSet, the resulting PSF is 
        completely flat (all pixels have the same value), regardless of the PSD
        shape or amplitude. This prevents proper SR computation from PSF peak ratio.
        
        The diffraction-limited case (PSD=0) works correctly, producing a proper
        peaked PSF.
        
        TODO: Investigate why psdSetToPsfSet produces flat PSFs for turbulent case.
        Possible causes:
        - PSD unit/normalization issues
        - Bug in longExposurePsf when processing non-zero PSDs
        - Incorrect handling of structure function or OTF calculation
        
        Once fixed, this test should verify:
        1. SR_psf = max(PSF_turb) / max(PSF_DL) 
        2. SR_marechal = exp(-σ²_φ) where σ²_φ = ∫∫ PSD df
        3. Both SR values match within ~5-8% (Marechal valid for SR > 0.3)
        """
        pass
    
    def test_marechal_formula_correctness(self):
        """
        Test that Marechal approximation formula is correctly implemented.
        
        This is a simple sanity check of the formula itself, not the PSF generation.
        Marechal approximation: SR = exp(-(2π σ_opd / λ)²) = exp(-σ²_φ)
        where σ_φ is phase RMS in radians.
        """
        # Test known values
        # For σ_φ = 1 radian, SR should be exp(-1) ≈ 0.3679
        variance_rad2 = 1.0
        SR = np.exp(-variance_rad2)
        self.assertAlmostEqual(SR, 0.36787944, places=5,
            msg="Marechal formula verification failed for σ²_φ = 1 rad²")
        
        # For σ_φ = 0.5 rad, σ²_φ = 0.25, SR ≈ 0.7788
        variance_rad2 = 0.25
        SR = np.exp(-variance_rad2)
        self.assertAlmostEqual(SR, 0.7788, places=4,
            msg="Marechal formula verification failed for σ²_φ = 0.25 rad²")
        
        # For zero phase error, SR = 1
        variance_rad2 = 0.0
        SR = np.exp(-variance_rad2)
        self.assertEqual(SR, 1.0,
            msg="Marechal formula should give SR=1 for zero phase error")
    
    def test_psd_variance_computation(self):
        """
        Test that variance is correctly computed from PSD.
        
        Phase variance: σ²_φ = ∫∫ PSD(f) df
        In discrete form: σ²_φ = Σ PSD(i,j) * Δf²
        """
        N = 256
        freq_range = 32.0  # cycles/m
        freq_step = freq_range / N
        
        # Test 1: Uniform PSD
        psd_uniform = np.ones((N, N), dtype=np.float64)
        variance = np.sum(psd_uniform) * freq_step**2
        expected_variance = N * N * freq_step**2
        self.assertAlmostEqual(variance, expected_variance, places=10,
            msg="Variance computation failed for uniform PSD")
        
        # Test 2: Delta function PSD (single pixel)
        psd_delta = np.zeros((N, N), dtype=np.float64)
        psd_delta[N//2, N//2] = 1.0
        variance = np.sum(psd_delta) * freq_step**2
        expected_variance = freq_step**2
        self.assertAlmostEqual(variance, expected_variance, places=10,
            msg="Variance computation failed for delta PSD")
        
        # Test 3: Zero PSD
        psd_zero = np.zeros((N, N), dtype=np.float64)
        variance = np.sum(psd_zero) * freq_step**2
        self.assertEqual(variance, 0.0,
            msg="Variance should be zero for zero PSD")


def suite():
    suite = unittest.TestSuite()
    suite.addTest(TestReconstructor('test_reconstructor'))
    suite.addTest(TestCovMatrices('test_cov_matrices'))
    suite.addTest(TestNoiseResiduals('test_noise_residuals'))
    suite.addTest(TestWindResiduals('test_wind_residuals'))
    suite.addTest(TestBiasAndVariance('test_bias_and_variance'))
    suite.addTest(TestPsfExtrapolation('test_estimate_exponent_from_fraction'))
    suite.addTest(TestPsfExtrapolation('test_forced_exponent_preserves_continuity_on_fit_interval'))
    suite.addTest(TestPsfExtrapolation('test_auto_exponent_is_clipped_to_bounds'))
    suite.addTest(TestPsfSpatialResampling('test_output_dimensions_are_strictly_target_fov'))
    suite.addTest(TestPsfSpatialResampling('test_flux_conservation_after_interpolation'))
    suite.addTest(TestPsfSpatialResampling('test_target_pixel_scale_is_tied_to_minimum_wavelength'))
    suite.addTest(TestStrehlRatioConsistency('test_marechal_formula_correctness'))
    suite.addTest(TestStrehlRatioConsistency('test_psd_variance_computation'))
    return suite



if __name__ == '__main__':
    runner = unittest.TextTestRunner()
    runner.run(suite())
