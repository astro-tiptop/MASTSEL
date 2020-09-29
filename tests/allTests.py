from mavis import *
import unittest


class TestReconstructor(unittest.TestCase):
    def test_reconstructor(self):
        """
        Test 
        """
        print("Running Test: TestReconstructor")
        cartPointingCoords = np.asarray([5,5])
        polarNGSCoords = np.asarray([[30.0,0.0], [50.0,100.0],[10.0,240.0]])
        cartNGSCoords = np.asarray([polarToCartesian(polarNGSCoords[0]), polarToCartesian(polarNGSCoords[1]), polarToCartesian(polarNGSCoords[2])])
        P_mat, rec_tomo, R_0, R_1 = buildReconstuctor(cartPointingCoords, cartNGSCoords)
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
        self.assertTrue( np.testing.assert_allclose(P_mat, P_mat__, rtol=1e-04, atol=1e-5)==None)
        self.assertTrue( np.testing.assert_allclose(rec_tomo, rec_tomo__, rtol=1e-04, atol=1e-5)==None)
        self.assertTrue( np.testing.assert_allclose(R_0, R_0__, rtol=1e-04, atol=1e-5)==None)
        self.assertTrue( np.testing.assert_allclose(R_1, R_1__, rtol=1e-04, atol=1e-5)==None)


class TestCovMatrices(unittest.TestCase):
    def test_cov_matrices(self):
        """
        Test 
        """
        cartPointingCoords = np.asarray([5,5])
        polarNGSCoords = np.asarray([[30.0,0.0], [50.0,100.0],[10.0,240.0]])
        cartNGSCoords = np.asarray([polarToCartesian(polarNGSCoords[0]), polarToCartesian(polarNGSCoords[1]), polarToCartesian(polarNGSCoords[2])])
        print("Running Test: TestCovMatrices")
        matCaaValue, matCasValue, matCssValue = computeCovMatrices(cartPointingCoords, cartNGSCoords)        
        hdul = fits.open('../data/Caa.fits')
        Caa_data = np.asarray(hdul[0].data, np.float64)
        hdul.close()
        hdul = fits.open('../data/Cas.fits')
        Cas_data = np.asarray(hdul[0].data, np.float64)
        hdul.close()
        hdul = fits.open('../data/Css.fits')
        Css_data = np.asarray(hdul[0].data, np.float64)
        hdul.close()
        self.assertTrue( np.testing.assert_allclose(Caa_data,matCaaValue, rtol=1e-02, atol=1e-5)==None)
        self.assertTrue( np.testing.assert_allclose(Cas_data,matCasValue, rtol=1e-02, atol=1e-5)==None)
        self.assertTrue( np.testing.assert_allclose(Css_data,matCssValue, rtol=1e-02, atol=1e-5)==None)        


class TestWindResiduals(unittest.TestCase):
    def test_wind_residuals(self):
        """
        Test 
        """ 
        print("Running Test: TestWindResiduals")
        NGS_flux = [10000, 30000, 5000]
        NGS_SR_1650 = [0.4, 0.2, 0.6]
        NGS_FWHM_mas = [90, 110, 85]
        mItGPU = Integrator('', cp, cp.float64)
        r1 = computeBias(NGS_flux[0], NGS_SR_1650[0], NGS_FWHM_mas[0], mItGPU)
        r2 = computeBias(NGS_flux[1], NGS_SR_1650[1], NGS_FWHM_mas[1], mItGPU)
        r3 = computeBias(NGS_flux[2], NGS_SR_1650[2], NGS_FWHM_mas[2], mItGPU)

        '''
        self.assertTrue( np.testing.assert_allclose( np.asarray(r1), np.asarray((0.4592354532951008, (0.1148088633237752, 3.348938898539153e-17), (0.08617446983322877, 0.08617446983322877))), rtol=1e-03, atol=1e-5)==None)
        self.assertTrue( np.testing.assert_allclose( np.asarray(r2), np.asarray( (0.43007711055063774, (0.10751927763765944, 2.0234128532071065e-17), (0.059932427347737786, 0.059932427347737786))), rtol=1e-03, atol=1e-5)==None)
        self.assertTrue( np.testing.assert_allclose( np.asarray(r3), np.asarray((0.42097905807645625, (0.10524476451911406, 0.0), (0.10381564644797976, 0.10381564644797978))), rtol=1e-03, atol=1e-5)==None)
        
        self.assertTrue( np.testing.assert_allclose( np.asarray(r1), np.asarray((0.4592354532951008, (0.1148088633237752, 3.348938898539153e-17), (0.08617446983322877, 0.08617446983322877))), rtol=1e-03, atol=1e-5)==None)
        self.assertTrue( np.testing.assert_allclose( np.asarray(r2), np.asarray( (0.43007711055063774, (0.10751927763765944, 2.0234128532071065e-17), (0.059932427347737786, 0.059932427347737786))), rtol=1e-03, atol=1e-5)==None)
        self.assertTrue( np.testing.assert_allclose( np.asarray(r3), np.asarray((0.42097905807645625, (0.10524476451911406, 0.0), (0.10381564644797976, 0.10381564644797978))), rtol=1e-03, atol=1e-5)==None)
        '''
        self.assertAlmostEqual(r1, (0.4592354532951008, (0.1148088633237752, 3.348938898539153e-17), (0.08617446983322877, 0.08617446983322877)))
        self.assertAlmostEqual(r2, (0.43007711055063774, (0.10751927763765944, 2.0234128532071065e-17), (0.059932427347737786, 0.059932427347737786)))
        self.assertAlmostEqual(r3, (0.42097905807645625, (0.10524476451911406, 0.0), (0.10381564644797976, 0.10381564644797978)) )


class TestNoiseResiduals(unittest.TestCase):
    def test_noise_residuals(self):
        """
        Test 
        """
        print("Running Test: TestNoiseResiduals")
        psd_freq, psd_tip_wind, psd_tilt_wind = loadWindPsd('../data/windpsd_mavis.fits')
        var1x = 0.05993281522281573 * pixel_scale**2
        bias = 0.4300779971881394
        nr = computeNoiseResidual(0.25, 250.0, 1000, var1x, bias, gpulib )
        wr = computeWindResidual(psd_freq, psd_tip_wind, psd_tilt_wind, var1x, bias, gpulib )
        result = nr[0]
        self.assertTrue( np.testing.assert_allclose(result, 2108.89544168, rtol=1e-03, atol=1e-5)==None)
        result = nr[1]
        self.assertTrue( np.testing.assert_allclose(result, 1361.65732465, rtol=1e-03, atol=1e-5)==None)
        result = wr[0]
        self.assertTrue( np.testing.assert_allclose(result, 71.89646223, rtol=1e-03, atol=1e-5)==None)
        result = wr[1]
        self.assertTrue( np.testing.assert_allclose(result, 61.02400116, rtol=1e-03, atol=1e-5)==None)
        
    
        
if __name__ == '__main__':
    unittest.main()