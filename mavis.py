from mavisUtilities import *
from mavisParams import *
from mavisFormulas import *

import multiprocessing as mp

import functools

fit_window_max_size = 512
defaultArrayBackend = cp

def hostData(_data):
    if defaultArrayBackend == cp:
        return cp.asnumpy(_data)
    else:
        return _data


def ft_ift2(G, xp=defaultArrayBackend):
    g = xp.fft.fftshift(xp.fft.ifft2(xp.fft.fftshift(G)))
    return g


def ft_ft2(G, xp=defaultArrayBackend):
    g = xp.fft.fftshift(xp.fft.fft2(xp.fft.fftshift(G)))
    return g


def FFTConvolve(in1, in2, xp=defaultArrayBackend):
    if in1.ndim == in2.ndim == 0:  # scalar inputs
        return in1 * in2
    elif not in1.ndim == in2.ndim:
        raise ValueError("Dimensions do not match.")
    elif in1.size == 0 or in2.size == 0:  # empty arrays
        return array([])
    rr = ft_ift2(ft_ft2(in1, xp) * ft_ft2(in2, xp), xp)
    return rr

# pixel size does not change


def zeroPad(input_grid, n, xp=defaultArrayBackend):
    N = input_grid.shape[0]
    output_grid = xp.zeros((N + 2 * n, N + 2 * n), dtype=input_grid.dtype)
    output_grid[n:N + n, n:N + n] = input_grid
    return output_grid

# pixel size does not change


def centralSquare(input_grid, n, xp=defaultArrayBackend):
    N = input_grid.shape[0]
    output_grid = xp.zeros((n, n), dtype=input_grid.dtype)
    output_grid = input_grid[int(N/2-n/2):int(N/2+n/2), int(N/2-n/2):int(N/2+n/2)]
    return output_grid


class Field(object):
    def __init__(
            self,
            wvl,
            N,
            width,
            unit='m',
            xp=defaultArrayBackend,
            _dtype=defaultArrayBackend.float64):
        self.xp = xp
        self.sampling = self.xp.zeros((N, N), dtype=_dtype)
        self.__N = N
        self.unit = unit
        self.width = width
        self.wvl = wvl

    @property
    def sampling(self):
        return self.__sampling

    @sampling.setter
    def sampling(self, value):
        self.__sampling = value
        self.__N = value.shape[0]

    @property
    def wvl(self):
        return self.__wvl

    @wvl.setter
    def wvl(self, value):
        self.__wvl = value
        self.__kk = 2.0 * np.pi / value / 1e9

    @property
    def kk(self):
        return self.__kk

    @property
    def N(self):
        return self.__N

    @property
    def width(self):
        return self.__width

    @width.setter
    def width(self, value):
        self.__width = value
        self.__pixel_size = self.__width / self.__N
        self.__half_pixel_size = self.__pixel_size / 2.0

    @property
    def pixel_size(self):
        return self.__pixel_size

    @property
    def half_pixel_size(self):
        return self.__half_pixel_size

    def hostData(self, _data):
        if self.xp == cp:
            return cp.asnumpy(_data)
        else:
            return _data

    def loadSamplingFromFile(self, filename):
        hdul = fits.open(filename)
        self.sampling = self.xp.asarray(hdul[0].data, self.sampling.dtype)

    # put the peak at the center of the Field, could change if needed
    # angle in rad
    # sigma_X, sigma_Y in the same unit as the Field
    def setAsGaussianKernel(self, sigma_X, sigma_Y, angle):
        gModel = models.Gaussian2D(
            amplitude=1,
            x_mean=self.N / 2,
            y_mean=self.N / 2,
            x_stddev=sigma_X / self.pixel_size,
            y_stddev=sigma_Y / self.pixel_size,
            theta=angle,
            cov_matrix=None)
        y, x = np.mgrid[:self.N, :self.N]
        hostSampling = gModel(x, y)
        self.sampling = self.xp.asarray(hostSampling)

    def setAsTelescopeMask(self, telescope_radius, occlusion_radius=0):
        fx = (self.xp.arange(-self.N / 2., self.N / 2., 1.0) + 0.5) * \
            self.pixel_size
        (fx, fy) = self.xp.meshgrid(fx, fx)
        self.sampling = self.xp.where(
            self.xp.logical_or(
                fx**2 +
                fy**2 > telescope_radius**2,
                fx**2 +
                fy**2 < occlusion_radius**2),
            0.0,
            1.0)
        self.telescope_radius = telescope_radius

    def setAsPSD(self, grid_diameter, r0_, L0_, l0_):
        grid_pixel_scale = grid_diameter / float(self.N)
        freq_range = 1.0 / grid_pixel_scale
        method = 'VonKarman'
        ss, new_pixel_size = ft_PSD_phi(
            r0_, self.N, freq_range, L0_, l0_, method)
        self.sampling = self.xp.asarray(ss)
        self.unit = 'm'
        self.width = new_pixel_size * self.N

    def PSDToPhaseScreen(self):
        #        complexField = xp.zeros( (N,N), dtype=xp.complex128 )
        #        rand_pahse = xp.random.random_sample( (N,N), dtype=xp.float64 )
        #        complexField = xp.exp( rand_pahse*1j*xp.pi*2.0 ) * xp.sqrt(xp.abs(self.sampling))
        #        realField = xp.real()
        N = self.N
        freq_range = self.width
        cn = (
            (self.xp.random.normal(
                size=(
                    N,
                    N)) +
                1j *
                self.xp.random.normal(
                size=(
                    N,
                    N))) *
            self.xp.sqrt(
                self.sampling) *
            self.pixel_size)
        self.sampling = ft_ift2(cn, self.xp).real
        self.width = 1.0 / (freq_range / N)

    def pupilToPsf(self):
        self.sampling = self.xp.square(
            self.xp.absolute(
                ft_ift2(
                    self.sampling,
                    self.xp)))
        in_grid_size = self.pixel_size * self.N
        new_pixel_size = self.wvl / in_grid_size
        self.unit = 'rad'
        self.width = new_pixel_size * self.N

    def pupilToOtf(self):
        self.sampling = self.xp.real(
            ft_ft2(
                self.xp.square(
                    self.xp.absolute(
                        ft_ift2(
                            self.sampling,
                            self.xp)))) )
    def FWHM(self):
        if self.N <= fit_window_max_size:
            s1 = self.sampling
        else:
            s1 = centralSquare(self.sampling, fit_window_max_size, self.xp)
        nn = min(fit_window_max_size, self.N)
        z = self.hostData(self.xp.copy(s1))
        y, x = np.mgrid[:nn, :nn]
        fit_p, p_init = fitGaussian(z)
        p = fit_p(p_init, x, y, z)
        sigma_X = p.x_stddev.value
        sigma_Y = p.y_stddev.value
        return (
            FWHM_from_sigma(sigma_X) *
            self.pixel_size,
            FWHM_from_sigma(sigma_Y) *
            self.pixel_size)

    def radial_profile(self, center):
        data = self.sampling
        xp = self.xp
        x, y = xp.indices((data.shape))
        r = xp.sqrt((x - center[0])**2 + (y - center[1])**2)
        r = r.astype(xp.int)
        tbin = xp.bincount(r.ravel(), data.ravel())
        nr = xp.bincount(r.ravel())
        radialprofile = tbin / nr
        return radialprofile

    def n_square_energy(self, center):
        data = self.sampling
        xp = self.xp
        x, y = xp.indices((data.shape))
        r = xp.fmin(xp.abs(x - center[0]), xp.abs(y - center[1]))
        r = r.astype(xp.int)
        tbin = xp.bincount(r.ravel(), data.ravel())
        nr = xp.bincount(r.ravel())
        radialprofile = tbin
        energy = xp.cumsum(radialprofile)
        return energy

    def peak(self):
        if self.N <= fit_window_max_size:
            s1 = self.sampling
        else:
            s1 = centralSquare(self.sampling, fit_window_max_size, self.xp)
        nn = min(fit_window_max_size, self.N)
        z = self.hostData(self.xp.copy(s1))
        y, x = np.mgrid[:nn, :nn]
        fit_p, p_init = fitGaussian(z)
        p = fit_p(p_init, x, y, z)
        value = p.amplitude.value
        return value

    def normalize(self):
        self.sampling = self.sampling / self.xp.sum(self.sampling)

    def downSample(self, factor):
        self.sampling = congrid(
            self.sampling, [
                self.N / factor, self.N / factor])
        self.width *= factor

    def standardPlot(self, log=False, zoom=1):
        img1 = self.xp.copy(self.sampling)
        if log:
            img1 = self.xp.log(self.xp.absolute(self.sampling))
        if zoom > 1:
            img1 = centralSquare(img1, int(self.N / zoom / 2), self.xp)

        img2 = self.hostData(img1)       
        standardPsfPlot(img2)

    def printStatus(self):
        print('Wavelength: ', self.wvl)
        print('Grid side elements: ', self.N)
        print('width: ', self.width, '[', self.unit, ']')
        if self.unit == 'rad':
            print('width: ', self.width * radiansToMas, '[ mas ]')
        print('pixel_size: ', self.pixel_size, '[', self.unit, ']')
        if self.unit == 'rad':
            print('pixel_size: ', self.pixel_size * radiansToMas, '[ mas ]')


def StrehlFromReference(psf, reference):
    pp = psf.peak()
    print("Actual psf peak:", pp)
    return pp / reference


def StrehlFromMask(psf, mask):
    refPsf = mask
    refPsf.pupilToPsf()
    ref = refPsf.peak()
    print("Diffraction limeted psf peak:", ref)
    return StrehlFromReference(psf, ref)


def convolve(psf, kernel, xp=defaultArrayBackend):
    xp = psf.xp
    if (psf.N != kernel.N):
        print(
            'psf and kernel sampling not compatible (grids sizes in pixels are different!)')
        return
    if xp.abs((psf.pixel_size - kernel.pixel_size)) > 0.001:
        print('These values should be the same!!!')
        print('psf pixel size: ', psf.pixel_size)
        print('kernel pixel size: ', kernel.pixel_size)
        return
    result = Field(psf.wvl, psf.N, psf.width, unit='m')

    if xp == cp:
        result.sampling = FFTConvolve(psf.sampling, kernel.sampling)
    else:
        result.sampling = scipy.signal.fftconvolve(
            psf.sampling, kernel.sampling)

    result.sampling = centralSquare(result.sampling, int(psf.N), xp)
    return result


def shortExposurePsf(mask, phaseScreen):
    xp = mask.xp
    if (mask.N != phaseScreen.N):
        print('Mask and phaseScreen sampling not compatible (grids sizes in pixels are different!)')
        return
    if xp.abs((mask.pixel_size - phaseScreen.pixel_size)) > 0.001:
        print('These values should be the same!!!')
        print('mask pixel size: ', mask.pixel_size)
        print('phaseScreen pixel size: ', phaseScreen.pixel_size)
        return
    result = Field(mask.wvl, mask.N, mask.width, unit='m')
    result.sampling = mask.sampling * phaseScreen.sampling
    result.standardPlot()
    result.pupilToPsf()
    return result


def longExposurePsf(mask, psd):
    xp = mask.xp
    if (mask.N != psd.N):
        print('Mask and PSD sampling not compatible (grids sizes in pixels are different!)')
        return
    freq_range = psd.width
    pitch = 1.0 / freq_range
    if xp.abs((mask.pixel_size - pitch) / pitch) > 0.001:
        print('These values should be the same!!!')
        print('otf_tel pixel size: ', mask.pixel_size)
        print('otf_turb pixel size: ', pitch)
        return
    p_final_psf = mask.wvl / (pitch * mask.N)  # rad
    result = Field(mask.wvl, mask.N, psd.N * p_final_psf, unit='rad')
    ################################################
    # step 0 : compute telescope otf
    mask.pupilToOtf()
    otf_tel = mask.sampling
    # step 1 : compute phase autocorrelation
    B_phi = xp.real(xp.fft.ifft2(xp.fft.ifftshift(psd.sampling))
                    ) * (psd.kk * freq_range) ** 2
    b0 = B_phi[0, 0]
    B_phi = xp.fft.fftshift(B_phi)
    # step 2 : compute structure function
    D_phi = 2.0 * (-B_phi + b0)
    # step 3 : compute turbolence otf
    otf_turb = xp.exp(-0.5 * (D_phi))
    # p_otft_turb = pitch
    # step 4 : combine telescope and turbolence otfs
    otf_system = otf_turb * otf_tel
    # step 5 : system otf to system psf
    result.sampling = xp.real(ft_ft2(otf_system))
    return result

# computation parameters

imax = 30
zmin = 0.03
zmax = 30
integrationPoints = 1000
psdIntegrationPoints = 4000
largeGridSize = 200
smallGridSize = 8
downsample_factor = 4
p_offset = 1.0 # 1/4 pixel on medium grid
mediumGridSize = int(largeGridSize/downsample_factor)
mediumShape = (mediumGridSize,mediumGridSize)
mediumPixelScale = pixel_scale/downsample_factor

# specialized formulas, mostly substituting parameter with mavisParametrs.py values

def specializedIM():
    apIM = mf['interactionMatrixNGS']
    apIM = subsParamsByName(apIM, {'D':TelescopeDiameter, 'r_FoV':MCAO_FoV*arcsecsToRadians/2.0, 'H_DM':DM_height})
    xx, yy = sp.symbols('x_1 y_1', real=True)
    apIM = subsParamsByName(apIM, {'x_NGS':xx, 'y_NGS':yy})
    apIM_func = sp.lambdify((xx, yy), apIM, modules=cpulib)
    return apIM, apIM_func


def specializedMeanVarFormulas(kind):
    dd0 = {'t':ThresholdWCoG, 'nu':NewValueThrPix, 'sigma_RON':sigmaRON}
    dd1 = {'b':Dark/SensorFrameRate}
    dd2 = {'F':ExcessNoiseFactor}
    expr0, exprK, integral = mf[kind]
    expr0 = subsParamsByName( expr0, {**dd0, **dd1} )
    exprK = subsParamsByName( exprK, {**dd1} )
    integral = subsParamsByName( integral,  {**dd0, **dd1, **dd2} )
    aFunction = exprK * integral.function
    return aFunction, expr0


def specializedTurbFuncs():
    aTurbPSDTip = subsParamsByName(mf['turbPSDTip'], {'V':WindSpeed, 'R':TelescopeDiameter/2.0, 'r_0':r0_Value, 'L_0':L0_Value, 'k_y_min':0.0001, 'k_y_max':100})
    aTurbPSDTilt = subsParamsByName(mf['turbPSDTilt'], {'V':WindSpeed, 'R':TelescopeDiameter/2.0, 'r_0':r0_Value, 'L_0':L0_Value, 'k_y_min':0.0001, 'k_y_max':100})
    return aTurbPSDTip, aTurbPSDTilt


def specializedC_coefficient():
    ffC = mf['noisePropagationCoefficient'].rhs
    fCValue1 = subsParamsByName(ffC, {'D':TelescopeDiameter, 'N_sa,tot':N_sa_tot })
    return fCValue1


def specializedNoiseFuncs():
    dict1 = {'d':loopDelaySteps, 'f_loop':SensorFrameRate}
    fTipS_LO1 = sp.simplify(subsParamsByName(mf['completeIntegralTipLO'], dict1 ).function)
    fTiltS_LO1 = sp.simplify(subsParamsByName(mf['completeIntegralTiltLO'], dict1).function)
    return fTipS_LO1, fTiltS_LO1


def specializedWindFuncs():
    dict1 = {'d':loopDelaySteps, 'f_loop':SensorFrameRate}
    fTipS1 = sp.simplify(subsParamsByName(mf['completeIntegralTip'], dict1).function)
    fTiltS1 = sp.simplify(subsParamsByName(mf['completeIntegralTilt'], dict1).function)
    return fTipS1, fTiltS1


zernikeCov_rh1 = MavisFormulas.getFormulaRhs('ZernikeCovarianceD')
zernikeCov_lh1 = MavisFormulas.getFormulaLhs('ZernikeCovarianceD')

def buildSpecializedCovFunctions():
    covValue_integrationLimits = (sp.symbols('f', positive=True), 1e-3, 10.0)
    cov_expr={}
    for li in [0,1,2]:
        paramDictBaseCov = { 'L_0': L0_Value, 'r_0': r0_Value, 'R_1': TelescopeDiameter/2.0, 'R_2': TelescopeDiameter/2.0, 'h': heights_Cn2[li] }
        for ii in [2,3]:
            for jj in [2,3]:
                cov_expr[ii+10*jj+100*li] = cov_expr_jk(zernikeCov_rh1, ii, jj)
                cov_expr[ii+10*jj+100*li] = subsParamsByName(cov_expr[ii+10*jj+100*li], paramDictBaseCov)
                cov_expr[ii+10*jj+100*li] = sp.Integral(cov_expr[ii+10*jj+100*li], covValue_integrationLimits)

    return cov_expr

# could use momoize instead, for now global
pIM, pIM_func = specializedIM()
aFunctionM, expr0M = specializedMeanVarFormulas('truncatedMeanComponents')
aFunctionV, expr0V = specializedMeanVarFormulas('truncatedVarianceComponents')
sTurbPSDTip, sTurbPSDTilt = specializedTurbFuncs()
fCValue = specializedC_coefficient()
fTipS_LO, fTiltS_LO = specializedNoiseFuncs()
fTipS, fTiltS = specializedWindFuncs()
specializedCovExprs = buildSpecializedCovFunctions()

# utility functions

def simple2Dgaussian(x, y, x0=0.0, y0=0.0, sg=1.0):
    return np.exp(-((x-x0)**2)/(2*sg**2)-((y-y0)**2)/(2*sg**2) )


def intRebin(arr, new_shape):
    shape = (new_shape[0], arr.shape[0] // new_shape[0],
             new_shape[1], arr.shape[1] // new_shape[1])
    return arr.reshape(shape).mean(-1).mean(1)


# numerical part
mItGPU = Integrator(cp, cp.float64, '')
mItCPU = Integrator(np, np.float64, '')

def buildReconstuctor(aCartPointingCoords, aCartNGSCoords):
    P, P_func = specializedIM()
    pp1 = P_func(aCartNGSCoords[0,0]*arcsecsToRadians, aCartNGSCoords[0,1]*arcsecsToRadians)
    pp2 = P_func(aCartNGSCoords[1,0]*arcsecsToRadians, aCartNGSCoords[1,1]*arcsecsToRadians)
    pp3 = P_func(aCartNGSCoords[2,0]*arcsecsToRadians, aCartNGSCoords[2,1]*arcsecsToRadians)
    P_mat = np.vstack([pp1, pp2, pp3]) # aka Interaction Matrix, im
    rec_tomo = scipy.linalg.pinv2(P_mat) # aka W, 5x6
    P_alpha0 = P_func(0, 0)
    P_alpha1 = P_func(aCartPointingCoords[0]*arcsecsToRadians, aCartPointingCoords[1]*arcsecsToRadians)
    R_0 = np.dot(P_alpha0, rec_tomo)
    R_1 = np.dot(P_alpha1, rec_tomo)
    return P_mat, rec_tomo, R_0, R_1


def compute2DMeanVar(aFunction, expr0, gaussianPointsM, smallGridSize, mIt):
    gaussianPoints = gaussianPointsM.reshape(smallGridSize*smallGridSize)
    aIntegral = sp.Integral(aFunction, (getSymbolByName(aFunction, 'z'), zmin, zmax), (getSymbolByName(aFunction, 'i'), 1, int(imax)) )
    paramsAndRanges = [( 'f_k', gaussianPoints, 0.0, 0.0, 'provided' )]
    lh = sp.Function('B')(getSymbolByName(aFunction, 'f_k'))
    xplot1, zplot1 = mIt.IntegralEval(lh, aIntegral, paramsAndRanges, [ (integrationPoints, 'linear'), (imax, 'linear')], 'raw')
    ssx, s0 = mIt.functionEval(expr0, paramsAndRanges )
    zplot1 = zplot1 + s0
    zplot1 = zplot1.reshape((smallGridSize,smallGridSize))
    return xplot1, zplot1


def meanVarSigma(gaussianPoints, smallGridSize, mIt):
    xplot1, mu_ktr_array = compute2DMeanVar( aFunctionM, expr0M, gaussianPoints, smallGridSize, mIt)
    xplot2, var_ktr_array = compute2DMeanVar( aFunctionV, expr0V, gaussianPoints, smallGridSize, mIt)
    var_ktr_array = var_ktr_array - mu_ktr_array**2
    sigma_ktr_array = np.sqrt(var_ktr_array.astype(np.float64))
    return mu_ktr_array, var_ktr_array, sigma_ktr_array


def computeBias(aNGS_flux, aNGS_SR_1650, aNGS_FWHM_mas, mIt):
    gridSpanArcsec= mediumPixelScale*largeGridSize/1000
    gridSpanRad = gridSpanArcsec/radiansToArcsecs
    peakValue = aNGS_flux/SensorFrameRate*aNGS_SR_1650*4.0*np.log(2)/(np.pi*(SensingWavelength/(TelescopeDiameter/2)*radiansToArcsecs*1000/mediumPixelScale)**2)
    peakValueNoFlux = aNGS_SR_1650*4.0*np.log(2)/(np.pi*(SensingWavelength/(TelescopeDiameter/2)*radiansToArcsecs*1000/mediumPixelScale)**2)
    xCoords=np.asarray(np.linspace(-largeGridSize/2.0+0.5, largeGridSize/2.0-0.5, largeGridSize), dtype=np.float64)
    yCoords=np.asarray(np.linspace(-largeGridSize/2.0+0.5, largeGridSize/2.0-0.5, largeGridSize), dtype=np.float64)
    xGrid, yGrid = np.meshgrid( xCoords, yCoords, sparse=False, copy=True)
    asigma = aNGS_FWHM_mas/sigmaToFWHM/mediumPixelScale
    g2d = peakValue * simple2Dgaussian( xGrid, yGrid, 0, 0, asigma)
    g2d = intRebin(g2d, mediumShape) * downsample_factor**2
    I_k_data = peakValue * simple2Dgaussian( xGrid, yGrid, 0, 0, asigma)
    I_k_prime_data = peakValue * simple2Dgaussian( xGrid, yGrid, p_offset, 0, asigma)
    back = skyBackground/SensorFrameRate
    I_k_data = intRebin(I_k_data, mediumShape) * downsample_factor**2
    I_k_prime_data = intRebin(I_k_prime_data,mediumShape) * downsample_factor**2
    f_k_data = I_k_data + back
    f_k_prime_data = I_k_prime_data + back
    W_Mask = np.zeros(mediumShape)
    ffx = np.arange(-mediumGridSize/2, mediumGridSize/2, 1.0) + 0.5
    (fx, fy) = np.meshgrid(ffx, ffx)
    W_Mask = np.where( np.logical_or(fx**2 +fy**2 > 2**2, fx**2 + fy**2 < 0**2), 0.0, 1.0)
    ii1, ii2 = int(mediumGridSize/2-smallGridSize/2), int(mediumGridSize/2+smallGridSize/2)
    f_k_data = f_k_data[ii1:ii2,ii1:ii2]
    f_k_prime_data = f_k_prime_data[ii1:ii2,ii1:ii2]
    W_Mask = W_Mask[ii1:ii2,ii1:ii2]
    fx = fx[ii1:ii2,ii1:ii2]
    fy = fy[ii1:ii2,ii1:ii2]
    gridSpanArcsec= mediumPixelScale*smallGridSize/1000
    gridSpanRad = gridSpanArcsec/radiansToArcsecs
    mu_ktr_array, var_ktr_array, sigma_ktr_array = meanVarSigma(f_k_data, smallGridSize, mIt)
    mu_ktr_prime_array, var_ktr_prime_array, sigma_ktr_prime_array = meanVarSigma(f_k_prime_data, smallGridSize, mIt)
    masked_mu0 = W_Mask*mu_ktr_array
    masked_mu = W_Mask*mu_ktr_prime_array
    masked_sigma = W_Mask*W_Mask*var_ktr_array
    mux = np.sum(masked_mu*fx)/np.sum(masked_mu)
    muy = np.sum(masked_mu*fy)/np.sum(masked_mu)
    varx = np.sum(masked_sigma*fx*fx)/(np.sum(masked_mu0)**2)
    vary = np.sum(masked_sigma*fy*fy)/(np.sum(masked_mu0)**2)
    bias = mux/(p_offset/downsample_factor)
    return (bias,(mux,muy),(varx,vary))


def computeWindPSDs(fmin, fmax, freq_samples):
    mIt = mItGPU
    paramAndRange = ( 'f', fmin, fmax, freq_samples, 'linear' )
    scaleFactor = 1000*np.pi/2.0  # from rad**2 to nm**2
    xplot1, zplot1 = mIt.IntegralEvalE(sTurbPSDTip, [paramAndRange], [(psdIntegrationPoints, 'linear')], 'rect')
    psd_freq = xplot1[0]
    psd_tip_wind = zplot1*scaleFactor
    xplot1, zplot1 = mIt.IntegralEvalE(sTurbPSDTilt, [paramAndRange], [(psdIntegrationPoints, 'linear')], 'rect')
    psd_tilt_wind = zplot1*scaleFactor
    return psd_tip_wind, psd_tilt_wind


def computeNoiseResidual(fmin, fmax, freq_samples, varX, bias, alib=gpulib):
    npoints = 99
    Cfloat = fCValue.evalf()
    psd_tip_wind, psd_tilt_wind = computeWindPSDs(fmin, fmax, freq_samples)
    psd_freq = np.asarray(np.linspace(fmin, fmax, freq_samples))
    df = psd_freq[1]-psd_freq[0]
    Df = psd_freq[-1]-psd_freq[0]
    sigma2Noise =  varX / bias**2 * Cfloat / (Df / df)
    # must wait till this moment to substitute the noise level
    fTipS1 = subsParamsByName(fTipS_LO, {'phi^noise_Tip': sigma2Noise})
    fTiltS1 = subsParamsByName( fTiltS_LO, {'phi^noise_Tilt': sigma2Noise})    
    fTipS_lambda1 = lambdifyByName( fTipS1, ['g^Tip_0', 'f', 'phi^wind_Tip'], alib)
    fTiltS_lambda1 = lambdifyByName( fTiltS1, ['g^Tilt_0', 'f', 'phi^wind_Tilt'], alib)
    if alib==gpulib:
        xp = cp
        psd_freq = cp.asarray(psd_freq)
        psd_tip_wind = cp.asarray(psd_tip_wind)
        psd_tilt_wind = cp.asarray(psd_tilt_wind)        
    else:
        xp = np
    g0g = xp.asarray( xp.linspace(0.01, 0.99, npoints) )
    e1 = psd_freq.reshape((1,psd_freq.shape[0]))
    e2 = psd_tip_wind.reshape((1,psd_tip_wind.shape[0]))
    e3 = psd_tilt_wind.reshape((1,psd_tilt_wind.shape[0]))
    e4 = g0g.reshape((g0g.shape[0], 1))
    psd_freq_ext, psd_tip_wind_ext, psd_tilt_wind_ext, g0g_ext = xp.broadcast_arrays(e1, e2, e3, e4)
    resultTip = xp.absolute((xp.sum(fTipS_lambda1( g0g_ext, psd_freq_ext, psd_tip_wind_ext), axis=(1)) ) )
    resultTilt = xp.absolute((xp.sum(fTiltS_lambda1( g0g_ext, psd_freq_ext, psd_tilt_wind_ext), axis=(1)) ) )
    minTipIdx = xp.where(resultTip == xp.amin(resultTip)) #    print(minTipIdx[0], resultTip[minTipIdx[0][0]])
    minTiltIdx = xp.where(resultTilt == xp.amin(resultTilt)) #    print(minTiltIdx[0], resultTilt[minTiltIdx[0][0]])
    return cp.asnumpy(resultTip[minTipIdx[0][0]]), cp.asnumpy(resultTilt[minTiltIdx[0][0]])


def computeWindResidual(psd_freq, psd_tip_wind0, psd_tilt_wind0, var1x, bias, alib=gpulib):
    npoints = 99
    Cfloat = fCValue.evalf()
    df = psd_freq[1]-psd_freq[0]
    Df = psd_freq[-1]-psd_freq[0]
    psd_tip_wind = psd_tip_wind0 * df
    psd_tilt_wind = psd_tilt_wind0 * df
    sigma2Noise = var1x / bias**2 * Cfloat / (Df / df)
    fTipS1 = subsParamsByName(fTipS, {'phi^noise_Tip': sigma2Noise})
    fTiltS1 = subsParamsByName( fTiltS, {'phi^noise_Tilt': sigma2Noise})
    fTipS_lambda1 = lambdifyByName( fTipS1, ['g^Tip_0', 'g^Tip_1', 'f', 'phi^wind_Tip'], alib)
    fTiltS_lambda1 = lambdifyByName( fTiltS1, ['g^Tilt_0', 'g^Tilt_1', 'f', 'phi^wind_Tilt'], alib)
    if alib==gpulib:
        xp = cp
        psd_freq = cp.asarray(psd_freq)
        psd_tip_wind = cp.asarray(psd_tip_wind)
        psd_tilt_wind = cp.asarray(psd_tilt_wind)        
    else:
        xp = np
    g0g, g1g = xp.meshgrid(xp.linspace(0.01,0.99,npoints), xp.linspace(0.01,0.99,npoints)) 
    e1 = psd_freq.reshape((1,1,psd_freq.shape[0]))
    e2 = psd_tip_wind.reshape((1,1,psd_tip_wind.shape[0]))
    e3 = psd_tilt_wind.reshape((1,1,psd_tilt_wind.shape[0]))
    e4 = g0g.reshape((g0g.shape[0],g0g.shape[1],1))
    e5 = g1g.reshape((g1g.shape[0],g1g.shape[1],1))
    psd_freq_ext, psd_tip_wind_ext, psd_tilt_wind_ext, g0g_ext, g1g_ext  = xp.broadcast_arrays(e1, e2, e3, e4, e5)
    resultTip = xp.absolute((xp.sum(fTipS_lambda1( g0g_ext, g1g_ext, psd_freq_ext, psd_tip_wind_ext), axis=(2)) ) )
    resultTilt = xp.absolute((xp.sum(fTiltS_lambda1( g0g_ext, g1g_ext, psd_freq_ext, psd_tilt_wind_ext), axis=(2)) ) )
    minTipIdx = xp.where(resultTip == xp.amin(resultTip))
    minTiltIdx = xp.where(resultTilt == xp.amin(resultTilt))
    return cp.asnumpy(resultTip[minTipIdx[0][0], minTipIdx[1][0]]), cp.asnumpy(resultTilt[minTiltIdx[0][0], minTiltIdx[1][0]])


def covValue(ii,jj, polarPointingCoordsDifference, li, mIt):
    paramsDict = {'theta': polarPointingCoordsDifference[1]*degToRad }
    _integral1 =  subsParamsByName(specializedCovExprs[ii+10*jj+100*li], paramsDict)
    rr0 = max( polarPointingCoordsDifference[0]*arcsecsToRadians, 1e-9)
    xplot1, zplot1 = mIt.IntegralEval(zernikeCov_lh1, _integral1, [('rho',  rr0, rr0, 1, 'linear')], [(integrationPoints, 'linear')], method='raw')
    return zplot1[0]


def computeCovMatrices( aCartPointingCoords, aCartNGSCoords):
    scaleF = (500.0/(2*np.pi))**2
    matCaaValue = np.zeros((2,2), dtype=np.float64)
    matCasValue = np.zeros((2,6), dtype=np.float64)
    matCssValue = np.zeros((6,6), dtype=np.float64)
    for ii in [2,3]:
        for jj in [2,3]:
            if ii!=jj:
                matCaaValue[ii-2][jj-2] = 0.0
            else:
                matCaaValue[ii-2][jj-2] = scaleF*covValue(ii,jj, cartesianToPolar(np.asarray([0.0,0.0])), 2, mItCPU)
    for li in [0,1]:
        for kk in [0,1,2]:
            polarPointingCoordsD = cartesianToPolar(aCartPointingCoords-aCartNGSCoords[kk,:])
            for ii in [2,3]:
                for jj in [2,3]:
                    matCasValue[ii-2][jj-2 + 2*kk] += Cn2_Value[li]*scaleF*covValue(ii, jj, polarPointingCoordsD, li, mItCPU)
    for li in [0,1]:
        for kk1 in [0,1,2]:
            for kk2 in [0,1,2]:
                polarPointingCoordsD = cartesianToPolar(aCartNGSCoords[kk1,:]-aCartNGSCoords[kk2,:])
                for ii in [2,3]:
                    for jj in [2,3]:
                        if kk1==kk2 and ii!=jj:
                            matCaaValue[ii-2][jj-2] = 0.0
                        else:
                            matCssValue[ii-2 + 2*kk1][jj-2 + 2*kk2] += Cn2_Value[li]*scaleF*covValue(ii,jj, polarPointingCoordsD, li, mItCPU)
    return matCaaValue, matCasValue, matCssValue


def loadWindPsd(filename):
    hdul = fits.open(filename)
    psd_data = np.asarray(hdul[0].data, np.float64)
    hdul.close()
    psd_freq = np.asarray(np.linspace(0.5, 250.0, 500))
    psd_tip_wind = np.zeros((500))
    psd_tilt_wind = np.zeros((500))
    psd_tip_wind[0:200] = psd_data[1,:]
    psd_tilt_wind[0:200] = psd_data[2,:]
    return psd_freq, psd_tip_wind, psd_tilt_wind


def CMatAssemble(aCartPointingCoordsV, aaCartNGSCoords, aCnn, aC1):
    P_mat, rec_tomo, R_0, R = buildReconstuctor(np.asarray(aCartPointingCoordsV), aaCartNGSCoords)
    Caa, Cas, Css = computeCovMatrices(np.asarray(aCartPointingCoordsV), aaCartNGSCoords)        
    RT = R.transpose()
    C2 = Caa + np.dot(R, np.dot(Css, RT)) - np.dot(Cas, RT) - np.dot(R, Cas.transpose())
    C3 = np.dot(R, np.dot(aCnn, RT))
#    print('R', R.shape, R)
#    print('Caa', Caa.shape, Caa)
#    print('Cas', Cas.shape, Cas)
#    print('Css', Css.shape, Css)
#    print('Cnn', aCnn.shape, aCnn)
#    print('C1', aC1.shape, aC1)
#    print('C2', C2.shape, C2)
#    print('C3', C3.shape, C3)   
    Ctot = aC1 + C2 + C3
    return Ctot


def computeTotalResidualMatrix(aCartPointingCoords, aCartNGSCoords, aNGS_flux, aNGS_SR_1650, aNGS_FWHM_mas, mono=False):
    nPointings = aCartPointingCoords.shape[0]
    psd_freq, psd_tip_wind, psd_tilt_wind = loadWindPsd('data/windpsd_mavis.fits')
    C1 = np.zeros((2,2))
    Cnn = np.zeros((6,6))
    maxFluxIndex = np.where(aNGS_flux==np.amax(aNGS_flux))
    for starIndex in [0,1,2]:
        bias, amu, avar = computeBias(aNGS_flux[starIndex], aNGS_SR_1650[starIndex], aNGS_FWHM_mas[starIndex], mItGPU) # one scalar, two tuples of 2
        var1x = avar[0] * pixel_scale**2
        nr = computeNoiseResidual(0.25, 250.0, 1000, var1x, bias, gpulib )
        wr = computeWindResidual(psd_freq, psd_tip_wind, psd_tilt_wind, var1x, bias, gpulib )
        Cnn[2*starIndex,2*starIndex] = nr[0]
        Cnn[2*starIndex+1,2*starIndex+1] = nr[1]
        if starIndex == maxFluxIndex[0][0]:
            C1[0,0] = wr[0]
            C1[1,1] = wr[1]
    # C1 and Cnn do not depend on aCartPointingCoords[i]
    if not mono:
        # while C2 and C3 do        
        inputs = aCartPointingCoords.tolist()
        pool_size = mp.cpu_count() * 2
        pool = mp.Pool(processes=pool_size)
        pool_outputs = pool.map(functools.partial(CMatAssemble, aaCartNGSCoords=aCartNGSCoords, aCnn=Cnn, aC1=C1) , inputs)
        pool.close()
        pool.join()
        return pool_outputs
    else:
        Ctot = np.zeros((nPointings,2,2))
        for i in range(nPointings):
            Ctot[i] = CMatAssemble(aCartPointingCoords[i], aCartNGSCoords, Cnn, C1)
        return Ctot


def ellipsesFromCovMats(Ctot):
    theta = sp.symbols('theta')
    sigma_1 = sp.symbols('sigma^2_1')
    sigma_2 = sp.symbols('sigma^2_2')
    sigma_X = sp.symbols('sigma^2_X')
    sigma_Y = sp.symbols('sigma^2_Y')
    sigma_XY = sp.symbols('sigma_XY')
    
    eq1 = sp.Eq(theta, sp.S(1)/sp.S(2) * sp.atan( 2*sigma_XY / ( sigma_X-sigma_Y )) )
    eq2 = sp.Eq(sigma_1, sp.S(1)/sp.S(2) * ( 2*sigma_XY / sp.sin(2*theta) + sigma_X + sigma_Y ) )
    eq3 = sp.Eq(sigma_2, sigma_X+sigma_Y-sigma_1 )

    leq1 = lambdifyByName( eq1.rhs, ['sigma^2_X', 'sigma^2_Y', 'sigma_XY'], cpulib)
    leq2 = lambdifyByName( eq2.rhs, ['sigma^2_X', 'sigma^2_Y', 'sigma_XY', 'theta'], cpulib)
    leq3 = lambdifyByName( eq3.rhs, ['sigma^2_X', 'sigma^2_Y', 'sigma^2_1'], cpulib)

    
    def computeCovEllispse(CC):
        th = leq1(CC[0,0], CC[1,1], CC[1,0])        
        s1 = leq2(CC[0,0], CC[1,1], CC[1,0], th)
        s2 = leq3(CC[0,0], CC[1,1], s1)
        if s2>=s1:
            th += np.pi/2.0
            smax = s2
            smin = s1
        else:
            smax = s1
            smin = s2
        scale = (np.pi/(180*3600*1000) * TelescopeDiameter / (4*1e-9))        
        return th, np.sqrt(smax)/scale, np.sqrt(smin)/scale

    
    def computeCovEllispses(Ctot):
        nPointings = Ctot.shape[0]
        result = np.zeros((nPointings, 3))
        covEllipses = np.zeros(nPointings)
        for i in range(nPointings):
            result[i] = computeCovEllispse(Ctot[i])
        return result

    return computeCovEllispses(Ctot)


import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse


def plotEllipses(cartPointingCoords, cov_ellipses, ss):
    nPointings = cov_ellipses.shape[0]
    ells = [Ellipse(xy=cartPointingCoords[i],  width=cov_ellipses[i,1]*ss, height=cov_ellipses[i,2]*ss, angle=cov_ellipses[i,0]*180/np.pi) for i in range(nPointings)]
    fig, ax = plt.subplots(subplot_kw={'aspect': 'equal'})
    for e in ells:
        ax.add_artist(e)
        e.set_clip_box(ax.bbox)
        #e.set_facecolor(np.random.rand(3))
    ax.set_xlim(-20, 20)
    ax.set_ylim(-20, 20)
    plt.show()


