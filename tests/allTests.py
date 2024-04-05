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


class TestWindResiduals(TestMavisLO):
    def test_wind_residuals(self):
        """
        Test 
        """ 
        print("Running Test: TestWindResiduals")
        NGS_flux = [2500, 7500 , 1250]
        NGS_freq = [500, 500 , 500]
        NGS_SR_1650 = [0.4, 0.2, 0.6]
        NGS_FWHM_mas = [51.677, 81.673, 42.373]
        
        mItGPU = Integrator(cp, cp.float64, '')
        r1 = TestMavisLO.mLO.computeBiasAndVariance(NGS_flux[0], NGS_freq[0], NGS_SR_1650[0], NGS_FWHM_mas[0])
        r2 = TestMavisLO.mLO.computeBiasAndVariance(NGS_flux[1], NGS_freq[1], NGS_SR_1650[1], NGS_FWHM_mas[1])
        r3 = TestMavisLO.mLO.computeBiasAndVariance(NGS_flux[2], NGS_freq[2], NGS_SR_1650[2], NGS_FWHM_mas[2])

        self.assertTrue( np.testing.assert_allclose(np.array(r1[0]), np.array((0.3305086490286356)), rtol=1e-03, atol=1e-5)==None)
        self.assertTrue( np.testing.assert_allclose(np.array(r1[1]), np.array((0.0826271622571589, 0.0)), rtol=1e-03, atol=1e-5)==None)
        self.assertTrue( np.testing.assert_allclose(np.array(r1[2]), np.array((0.12942792320851965, 0.12942792320851965)), rtol=1e-03, atol=1e-5)==None)
        self.assertTrue( np.testing.assert_allclose(np.array(r2[0]), np.array((0.3067767436965856)), rtol=1e-03, atol=1e-5)==None)
        self.assertTrue( np.testing.assert_allclose(np.array(r2[1]), np.array((0.0766941859241464, 0.0)), rtol=1e-03, atol=1e-5)==None)
        self.assertTrue( np.testing.assert_allclose(np.array(r2[2]), np.array((0.1381058202766471, 0.1381058202766471)), rtol=1e-03, atol=1e-5)==None)
        self.assertTrue( np.testing.assert_allclose(np.array(r3[0]), np.array((0.28524614382590824)), rtol=1e-03, atol=1e-5)==None)
        self.assertTrue( np.testing.assert_allclose(np.array(r3[1]), np.array((0.07131153595647706, 0.0)), rtol=1e-03, atol=1e-5)==None)
        self.assertTrue( np.testing.assert_allclose(np.array(r3[2]), np.array((0.13778324257626204, 0.13778324257626204)), rtol=1e-03, atol=1e-5)==None)


class TestNoiseResiduals(TestMavisLO):
    def test_noise_residuals(self):
        """
        Test 
        """
        print("Running Test: TestNoiseResiduals")
        psd_freq, psd_tip_wind, psd_tilt_wind = TestMavisLO.mLO.loadWindPsd('data/windpsd_mavis.fits')
        var1x = 0.05993281522281573 * TestMavisLO.mLO.PixelScale_LO**2
        bias = 0.4300779971881394
        nr = TestMavisLO.mLO.computeNoiseResidual(0.25, 250.0, 1000, var1x, bias, gpulib )
        wr = TestMavisLO.mLO.computeWindResidual(psd_freq, psd_tip_wind, psd_tilt_wind, var1x, bias, gpulib )
        result = nr[0]
        self.assertTrue( np.testing.assert_allclose(result, 3827.13, rtol=1e-03, atol=1e-5)==None)
        result = nr[1]
        self.assertTrue( np.testing.assert_allclose(result, 2387.49, rtol=1e-03, atol=1e-5)==None)
        result = wr[0]
        self.assertTrue( np.testing.assert_allclose(result, 243.63, rtol=1e-03, atol=1e-5)==None)
        result = wr[1]
        self.assertTrue( np.testing.assert_allclose(result, 173.69, rtol=1e-03, atol=1e-5)==None)



def suite():
    suite = unittest.TestSuite()
    suite.addTest(TestReconstructor('test_reconstructor'))
    suite.addTest(TestCovMatrices('test_cov_matrices'))
    suite.addTest(TestWindResiduals('test_wind_residuals'))
    suite.addTest(TestNoiseResiduals('test_noise_residuals'))
    return suite



if __name__ == '__main__':
    runner = unittest.TextTestRunner()
    runner.run(suite())
