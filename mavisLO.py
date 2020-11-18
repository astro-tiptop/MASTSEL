from mavisUtilities import *
from mavisFormulas import *
# import multiprocessing as mp
import functools
from configparser import ConfigParser

class MavisLO(object):
    
    def __init__(self, path, parametersFile, windpsdfile):
                
        parser = ConfigParser()
        parser.read(path + parametersFile + '.ini')
#
# SETTING PARAMETERS READ FROM FILE       
#
# parameters which are not used in this class reported but commented
        self.TelescopeDiameter   = eval(parser.get('telescope', 'TelescopeDiameter'))
#        self.zenithAngle         = zenithAngle
#        self.obscurationRatio    = obscurationRatio
#        self.resolution          = resolution
#        self.path_pupil          = path_pupil
        self.atmosphereWavelength= eval(parser.get('atmosphere', 'atmosphereWavelength'))
        self.seeing              = eval(parser.get('atmosphere', 'seeing'))
        self.L0                  = eval(parser.get('atmosphere', 'L0'))
        self.Cn2Weights          = eval(parser.get('atmosphere', 'Cn2Weights'))
        self.Cn2Heights          = eval(parser.get('atmosphere', 'Cn2Heights'))
        self.Cn2RefHeight          = eval(parser.get('atmosphere', 'Cn2RefHeight'))        
        self.wSpeed              = eval(parser.get('atmosphere', 'wSpeed'))
#        self.wDir                = wDir
#        self.nLayersReconstructed= nLayersReconstructed
#        self.ScienceWavelength   = ScienceWavelength
#        self.ScienceZenith       = ScienceZenith
#        self.ScienceAzimuth      = ScienceAzimuth
#        self.psInMas             = psInMas
#        self.psf_FoV             = psf_FoV
        self.technical_FoV       = eval(parser.get('PSF_DIRECTIONS', 'technical_FoV'))
#        self.GuideStarZenith_HO  = GuideStarZenith_HO
#        self.GuideStarAzimuth_HO = GuideStarAzimuth_HO
#        self.GuideStarHeight_HO  = GuideStarHeight_HO
#        self.DmPitchs            = DmPitchs
        self.DmHeights           = eval(parser.get('DM', 'DmHeights'))
#        self.OptimizationZenith  = OptimizationZenith
#        self.OptimizationAzimuth = OptimizationAzimuth
#        self.OptimizationWeight  = OptimizationWeight
#        self.OptimizationConditioning = OptimizationConditioning
#        self.nLenslet_HO            = nLenslet_HO
#        self.SensingWavelength_HO   = SensingWavelength_HO
#        self.loopGain_HO            = loopGain_HO
#        self.SensorFrameRate_HO     = SensorFrameRate_HO
#        self.loopDelaySteps_HO      = loopDelaySteps_HO
#        self.nph_HO                 = nph_HO
#        self.sigmaRON_HO            = sigmaRON_HO
#        self.Npix_per_subap_HO      = Npix_per_subap_HO
#        self.pixel_scale_HO         = pixel_scale_HO
        self.N_sa_tot_LO            = eval(parser.get('SENSOR_LO', 'N_sa_tot_LO'))
        self.SensingWavelength_LO   = eval(parser.get('SENSOR_LO', 'SensingWavelength_LO'))
        self.SensorFrameRate_LO     = eval(parser.get('SENSOR_LO', 'SensorFrameRate_LO'))
        self.loopDelaySteps_LO      = eval(parser.get('SENSOR_LO', 'loopDelaySteps_LO'))
        self.pixel_scale_LO         = eval(parser.get('SENSOR_LO', 'pixel_scale_LO'))
#        self.Npix_per_subap_LO      = Npix_per_subap_LO
        self.WindowRadiusWCoG_LO    = eval(parser.get('SENSOR_LO', 'WindowRadiusWCoG_LO'))
        self.sigmaRON_LO            = eval(parser.get('SENSOR_LO', 'sigmaRON_LO'))
        self.ExcessNoiseFactor_LO   = eval(parser.get('SENSOR_LO', 'ExcessNoiseFactor_LO'))
        self.Dark_LO                = eval(parser.get('SENSOR_LO', 'Dark_LO'))
        self.skyBackground_LO       = eval(parser.get('SENSOR_LO', 'skyBackground_LO'))
        self.ThresholdWCoG_LO       = eval(parser.get('SENSOR_LO', 'ThresholdWCoG_LO'))
        self.NewValueThrPix_LO      = eval(parser.get('SENSOR_LO', 'NewValueThrPix_LO'))
#
# END OF SETTING PARAMETERS READ FROM FILE       
#

        vr0 = eval(parser.get('atmosphere', 'r0_Value'))
        if vr0:
            self.r0_Value = vr0
        else:
            self.r0_Value = 0.976*self.atmosphereWavelength/self.seeing*206264.8 # old: 0.15        
        #  print(self.r0_Value) # 0.1677620373333333

        vSpeed = eval(parser.get('atmosphere', 'oneWindSpeed'))
        if vr0:
            self.WindSpeed = vSpeed
        else:
            self.WindSpeed = (np.dot( np.power(np.asarray(self.wSpeed), 5.0/3.0), np.asarray(self.Cn2Weights) ) / np.sum( np.asarray(self.Cn2Weights) ) ) ** (3.0/5.0)
        # print('WindSpeed', self.WindSpeed) # result is 11.94
        
#        self.mutex = None
        self.imax = 30
        self.zmin = 0.03
        self.zmax = 30
        self.integrationPoints = 1000 # //2
        self.psdIntegrationPoints = 4000 # //2
        self.largeGridSize = 200
        self.downsample_factor = 4
        self.smallGridSize = 2*self.WindowRadiusWCoG_LO
        self.p_offset = 1.0 # 1/4 pixel on medium grid
        self.mediumGridSize = int(self.largeGridSize/self.downsample_factor)
        self.mediumShape = (self.mediumGridSize,self.mediumGridSize)
        self.mediumPixelScale = self.pixel_scale_LO/self.downsample_factor
        self.zernikeCov_rh1 = MavisFormulas.getFormulaRhs('ZernikeCovarianceD')
        self.zernikeCov_lh1 = MavisFormulas.getFormulaLhs('ZernikeCovarianceD')
        self.sTurbPSDTip, self.sTurbPSDTilt = self.specializedTurbFuncs()
        self.fCValue = self.specializedC_coefficient()
        self.fTipS_LO, self.fTiltS_LO = self.specializedNoiseFuncs()
        self.fTipS, self.fTiltS = self.specializedWindFuncs()
        self.specializedCovExprs = self.buildSpecializedCovFunctions()
        self.aFunctionM, self.expr0M = self.specializedMeanVarFormulas('truncatedMeanComponents')
        self.aFunctionV, self.expr0V = self.specializedMeanVarFormulas('truncatedVarianceComponents')

        self.mItGPU = Integrator(cp, cp.float64, '')
        self.mItGPUcomplex = Integrator(cp, cp.complex64, '')
        self.psd_freq, self.psd_tip_wind, self.psd_tilt_wind = self.loadWindPsd(windpsdfile)


    # specialized formulas, mostly substituting parameter with mavisParametrs.py values
    def specializedIM(self, alib=cpulib):
        apIM = mf['interactionMatrixNGS']
        apIM = subsParamsByName(apIM, {'D':self.TelescopeDiameter, 'r_FoV':self.technical_FoV*arcsecsToRadians/2.0, 'H_DM':max(self.DmHeights)})
        xx, yy = sp.symbols('x_1 y_1', real=True)
        apIM = subsParamsByName(apIM, {'x_NGS':xx, 'y_NGS':yy})
        apIM_func = sp.lambdify((xx, yy), apIM, modules=alib)
        return apIM, apIM_func

    
    def specializedMeanVarFormulas(self, kind):
        dd0 = {'t':self.ThresholdWCoG_LO, 'nu':self.NewValueThrPix_LO, 'sigma_RON':self.sigmaRON_LO}
        dd1 = {'b':self.Dark_LO/self.SensorFrameRate_LO}
        dd2 = {'F':self.ExcessNoiseFactor_LO}
        expr0, exprK, integral = mf[kind+"0"], mf[kind+"1"], mf[kind+"2"]
        expr0 = subsParamsByName( expr0, {**dd0, **dd1} )
        exprK = subsParamsByName( exprK, {**dd1} )
        integral = subsParamsByName( integral,  {**dd0, **dd1, **dd2} )
        aFunction = exprK * integral.function
        return aFunction, expr0
    
    
    def specializedTurbFuncs(self):
        aTurbPSDTip = subsParamsByName(mf['turbPSDTip'], {'V':self.WindSpeed, 'R':self.TelescopeDiameter/2.0, 'r_0':self.r0_Value, 'L_0':self.L0, 'k_y_min':0.0001, 'k_y_max':100})
        aTurbPSDTilt = subsParamsByName(mf['turbPSDTilt'], {'V':self.WindSpeed, 'R':self.TelescopeDiameter/2.0, 'r_0':self.r0_Value, 'L_0':self.L0, 'k_y_min':0.0001, 'k_y_max':100})
        return aTurbPSDTip, aTurbPSDTilt

    
    def specializedC_coefficient(self):
        ffC = mf['noisePropagationCoefficient'].rhs
        self.fCValue1 = subsParamsByName(ffC, {'D':self.TelescopeDiameter, 'N_sa_tot':self.N_sa_tot_LO })
        return self.fCValue1

    
    def specializedNoiseFuncs(self):
        dict1 = {'d':self.loopDelaySteps_LO, 'f_loop':self.SensorFrameRate_LO}
        self.fTipS_LO1 = sp.simplify(subsParamsByName(mf['completeIntegralTipLO'], dict1 ).function)
        self.fTiltS_LO1 = sp.simplify(subsParamsByName(mf['completeIntegralTiltLO'], dict1).function)
        return self.fTipS_LO1, self.fTiltS_LO1

    
    def specializedWindFuncs(self):
        dict1 = {'d':self.loopDelaySteps_LO, 'f_loop':self.SensorFrameRate_LO}
        self.fTipS1 = sp.simplify(subsParamsByName(mf['completeIntegralTip'], dict1).function)
        self.fTiltS1 = sp.simplify(subsParamsByName(mf['completeIntegralTilt'], dict1).function)
        return self.fTipS1, self.fTiltS1
    
    
    def buildSpecializedCovFunctions(self):
        covValue_integrationLimits = (sp.symbols('f', positive=True), 1e-3, 10.0)
        p = sp.symbols('p', real=False)
        cov_expr={}
        paramDictBaseCov = { 'L_0': self.L0, 'r_0': self.r0_Value, 'R_1': self.TelescopeDiameter/2.0, 'R_2': self.TelescopeDiameter/2.0} 
        for ii in [2,3]:
            for jj in [2,3]:
                aa = subsParamsByName(cov_expr_jk(self.zernikeCov_rh1, ii, jj), paramDictBaseCov)
                aaint = sp.Integral(aa, covValue_integrationLimits)
                aaint = subsParamsByName(aaint, {'rho': sp.Abs(p), 'theta': sp.arg(p)} )
                cov_expr[ii+10*jj] = aaint

        return cov_expr

    
    def buildReconstuctor(self, aCartPointingCoords, aCartNGSCoords):
        P, P_func = self.specializedIM()
        pp1 = P_func(aCartNGSCoords[0,0]*arcsecsToRadians, aCartNGSCoords[0,1]*arcsecsToRadians)
        pp2 = P_func(aCartNGSCoords[1,0]*arcsecsToRadians, aCartNGSCoords[1,1]*arcsecsToRadians)
        pp3 = P_func(aCartNGSCoords[2,0]*arcsecsToRadians, aCartNGSCoords[2,1]*arcsecsToRadians)
        P_mat = np.vstack([pp1, pp2, pp3]) # aka Interaction Matrix, im
        rec_tomo = scipy.linalg.pinv(P_mat) # aka W, 5x6
        P_alpha0 = P_func(0, 0)
        P_alpha1 = P_func(aCartPointingCoords[0]*arcsecsToRadians, aCartPointingCoords[1]*arcsecsToRadians)
        R_0 = np.dot(P_alpha0, rec_tomo)
        R_1 = np.dot(P_alpha1, rec_tomo)
        return P_mat, rec_tomo, R_0, R_1

    
    def buildReconstuctor2(self, aCartPointingCoordsV, aCartNGSCoords):
        npointings = aCartPointingCoordsV.shape[0]
        P, P_func = self.specializedIM()
        pp1 = P_func(aCartNGSCoords[0,0]*arcsecsToRadians, aCartNGSCoords[0,1]*arcsecsToRadians)
        pp2 = P_func(aCartNGSCoords[1,0]*arcsecsToRadians, aCartNGSCoords[1,1]*arcsecsToRadians)
        pp3 = P_func(aCartNGSCoords[2,0]*arcsecsToRadians, aCartNGSCoords[2,1]*arcsecsToRadians)
        P_mat = np.vstack([pp1, pp2, pp3]) # aka Interaction Matrix, im
        rec_tomo = scipy.linalg.pinv(P_mat) # aka W, 5x6
        vx = np.asarray(aCartPointingCoordsV[:,0])
        vy = np.asarray(aCartPointingCoordsV[:,1])
        R_1 = np.zeros((2*npointings, 6))
        for k in range(npointings):
            P_alpha1 = P_func(vx[k]*arcsecsToRadians, vy[k]*arcsecsToRadians)
            R_1[2*k:2*(k+1), :] = cp.dot(P_alpha1, rec_tomo)
        return R_1, R_1.transpose()

    
    def compute2DMeanVar(self, aFunction, expr0, gaussianPointsM):
        gaussianPoints = gaussianPointsM.reshape(self.smallGridSize*self.smallGridSize)
        aIntegral = sp.Integral(aFunction, (getSymbolByName(aFunction, 'z'), self.zmin, self.zmax), (getSymbolByName(aFunction, 'i'), 1, int(self.imax)) )
        paramsAndRanges = [( 'f_k', gaussianPoints, 0.0, 0.0, 'provided' )]
        lh = sp.Function('B')(getSymbolByName(aFunction, 'f_k'))
        xplot1, zplot1 = self.mItGPU.IntegralEval(lh, aIntegral, paramsAndRanges, [ (self.integrationPoints//2, 'linear'), (self.imax, 'linear')], 'raw')
        ssx, s0 = self.mItGPU.functionEval(expr0, paramsAndRanges )
        zplot1 = zplot1 + s0
        zplot1 = zplot1.reshape((self.smallGridSize,self.smallGridSize))
        return xplot1, zplot1

    
    def meanVarSigma(self, gaussianPoints):
        xplot1, mu_ktr_array = self.compute2DMeanVar( self.aFunctionM, self.expr0M, gaussianPoints)
        xplot2, var_ktr_array = self.compute2DMeanVar( self.aFunctionV, self.expr0V, gaussianPoints)
        var_ktr_array = var_ktr_array - mu_ktr_array**2
        sigma_ktr_array = np.sqrt(var_ktr_array.astype(np.float32))
        return mu_ktr_array, var_ktr_array, sigma_ktr_array

        
    def computeBias(self, aNGS_flux, aNGS_SR_1650, aNGS_FWHM_mas):
        gridSpanArcsec= self.mediumPixelScale*self.largeGridSize/1000
        gridSpanRad = gridSpanArcsec/radiansToArcsecs
        peakValue = aNGS_flux/self.SensorFrameRate_LO*aNGS_SR_1650*4.0*np.log(2)/(np.pi*(self.SensingWavelength_LO/(self.TelescopeDiameter/2)*radiansToArcsecs*1000/self.mediumPixelScale)**2)
        peakValueNoFlux = aNGS_SR_1650*4.0*np.log(2)/(np.pi*(self.SensingWavelength_LO/(self.TelescopeDiameter/2)*radiansToArcsecs*1000/self.mediumPixelScale)**2)
        xCoords=np.asarray(np.linspace(-self.largeGridSize/2.0+0.5, self.largeGridSize/2.0-0.5, self.largeGridSize), dtype=np.float32)
        yCoords=np.asarray(np.linspace(-self.largeGridSize/2.0+0.5, self.largeGridSize/2.0-0.5, self.largeGridSize), dtype=np.float32)
        xGrid, yGrid = np.meshgrid( xCoords, yCoords, sparse=False, copy=True)
        asigma = aNGS_FWHM_mas/sigmaToFWHM/self.mediumPixelScale
        g2d = peakValue * simple2Dgaussian( xGrid, yGrid, 0, 0, asigma)
        g2d = intRebin(g2d, self.mediumShape) * self.downsample_factor**2
        I_k_data = peakValue * simple2Dgaussian( xGrid, yGrid, 0, 0, asigma)
        I_k_prime_data = peakValue * simple2Dgaussian( xGrid, yGrid, self.p_offset, 0, asigma)
        back = self.skyBackground_LO/self.SensorFrameRate_LO
        I_k_data = intRebin(I_k_data, self.mediumShape) * self.downsample_factor**2
        I_k_prime_data = intRebin(I_k_prime_data,self.mediumShape) * self.downsample_factor**2
        f_k_data = I_k_data + back
        f_k_prime_data = I_k_prime_data + back
        W_Mask = np.zeros(self.mediumShape)
        ffx = np.arange(-self.mediumGridSize/2, self.mediumGridSize/2, 1.0) + 0.5
        (fx, fy) = np.meshgrid(ffx, ffx)
        W_Mask = np.where( np.logical_or(fx**2 +fy**2 > self.WindowRadiusWCoG_LO**2, fx**2 + fy**2 < 0**2), 0.0, 1.0)
        ii1, ii2 = int(self.mediumGridSize/2-self.smallGridSize/2), int(self.mediumGridSize/2+self.smallGridSize/2)
        f_k_data = f_k_data[ii1:ii2,ii1:ii2]
        f_k_prime_data = f_k_prime_data[ii1:ii2,ii1:ii2]
        W_Mask = W_Mask[ii1:ii2,ii1:ii2]
        fx = fx[ii1:ii2,ii1:ii2]
        fy = fy[ii1:ii2,ii1:ii2]
        gridSpanArcsec= self.mediumPixelScale*self.smallGridSize/1000
        gridSpanRad = gridSpanArcsec/radiansToArcsecs
        mu_ktr_array, var_ktr_array, sigma_ktr_array = self.meanVarSigma(f_k_data)
        mu_ktr_prime_array, var_ktr_prime_array, sigma_ktr_prime_array = self.meanVarSigma(f_k_prime_data)
        masked_mu0 = W_Mask*mu_ktr_array
        masked_mu = W_Mask*mu_ktr_prime_array
        masked_sigma = W_Mask*W_Mask*var_ktr_array
        mux = np.sum(masked_mu*fx)/np.sum(masked_mu)
        muy = np.sum(masked_mu*fy)/np.sum(masked_mu)
        varx = np.sum(masked_sigma*fx*fx)/(np.sum(masked_mu0)**2)
        vary = np.sum(masked_sigma*fy*fy)/(np.sum(masked_mu0)**2)
        bias = mux/(self.p_offset/self.downsample_factor)
        return (bias,(mux,muy),(varx,vary))

    
    def computeWindPSDs(self, fmin, fmax, freq_samples):    
        paramAndRange = ( 'f', fmin, fmax, freq_samples, 'linear' )
        scaleFactor = 1000*np.pi/2.0  # from rad**2 to nm**2
        xplot1, zplot1 = self.mItGPU.IntegralEvalE(self.sTurbPSDTip, [paramAndRange], [(self.psdIntegrationPoints, 'linear')], 'rect')
        psd_freq = xplot1[0]
        psd_tip_wind = zplot1*scaleFactor
        xplot1, zplot1 = self.mItGPU.IntegralEvalE(self.sTurbPSDTilt, [paramAndRange], [(self.psdIntegrationPoints, 'linear')], 'rect')
        psd_tilt_wind = zplot1*scaleFactor
        return psd_tip_wind, psd_tilt_wind

        
    def computeNoiseResidual(self, fmin, fmax, freq_samples, varX, bias, alib=gpulib):
        npoints = 99
        Cfloat = self.fCValue.evalf()
        psd_tip_wind, psd_tilt_wind = self.computeWindPSDs(fmin, fmax, freq_samples)
        psd_freq = np.asarray(np.linspace(fmin, fmax, freq_samples))
        df = psd_freq[1]-psd_freq[0]
        Df = psd_freq[-1]-psd_freq[0]
        sigma2Noise =  varX / bias**2 * Cfloat / (Df / df)
        # must wait till this moment to substitute the noise level
        self.fTipS1 = subsParamsByName(self.fTipS_LO, {'phi^noise_Tip': sigma2Noise})
        self.fTiltS1 = subsParamsByName( self.fTiltS_LO, {'phi^noise_Tilt': sigma2Noise})    
        self.fTipS_lambda1 = lambdifyByName( self.fTipS1, ['g^Tip_0', 'f', 'phi^wind_Tip'], alib)
        self.fTiltS_lambda1 = lambdifyByName( self.fTiltS1, ['g^Tilt_0', 'f', 'phi^wind_Tilt'], alib)
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
        resultTip = xp.absolute((xp.sum(self.fTipS_lambda1( g0g_ext, psd_freq_ext, psd_tip_wind_ext), axis=(1)) ) )
        resultTilt = xp.absolute((xp.sum(self.fTiltS_lambda1( g0g_ext, psd_freq_ext, psd_tilt_wind_ext), axis=(1)) ) )
        minTipIdx = xp.where(resultTip == xp.amin(resultTip)) #    print(minTipIdx[0], resultTip[minTipIdx[0][0]])
        minTiltIdx = xp.where(resultTilt == xp.amin(resultTilt)) #    print(minTiltIdx[0], resultTilt[minTiltIdx[0][0]])
        return cp.asnumpy(resultTip[minTipIdx[0][0]]), cp.asnumpy(resultTilt[minTiltIdx[0][0]])

        
    def computeWindResidual(self, psd_freq, psd_tip_wind0, psd_tilt_wind0, var1x, bias, alib=gpulib):
        npoints = 99
        Cfloat = self.fCValue.evalf()
        df = psd_freq[1]-psd_freq[0]
        Df = psd_freq[-1]-psd_freq[0]
        psd_tip_wind = psd_tip_wind0 * df
        psd_tilt_wind = psd_tilt_wind0 * df
        sigma2Noise = var1x / bias**2 * Cfloat / (Df / df)
        self.fTipS1 = subsParamsByName(self.fTipS, {'phi^noise_Tip': sigma2Noise})
        self.fTiltS1 = subsParamsByName( self.fTiltS, {'phi^noise_Tilt': sigma2Noise})
        self.fTipS_lambda1 = lambdifyByName( self.fTipS1, ['g^Tip_0', 'g^Tip_1', 'f', 'phi^wind_Tip'], alib)
        self.fTiltS_lambda1 = lambdifyByName( self.fTiltS1, ['g^Tilt_0', 'g^Tilt_1', 'f', 'phi^wind_Tilt'], alib)
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
        resultTip = xp.absolute((xp.sum(self.fTipS_lambda1( g0g_ext, g1g_ext, psd_freq_ext, psd_tip_wind_ext), axis=(2)) ) )
        resultTilt = xp.absolute((xp.sum(self.fTiltS_lambda1( g0g_ext, g1g_ext, psd_freq_ext, psd_tilt_wind_ext), axis=(2)) ) )
        minTipIdx = xp.where(resultTip == xp.amin(resultTip))
        minTiltIdx = xp.where(resultTilt == xp.amin(resultTilt))
        return cp.asnumpy(resultTip[minTipIdx[0][0], minTipIdx[1][0]]), cp.asnumpy(resultTilt[minTiltIdx[0][0], minTiltIdx[1][0]])

        
    def covValue(self, ii,jj, pp, hh):
        p =sp.symbols('p', real=False)
        h =sp.symbols('h', positive=True)
#    with self.mutex:
        xplot1, zplot1 = self.mItGPUcomplex.IntegralEval(sp.Function('C_v')(p, h), 
                                                         self.specializedCovExprs[ii+10*jj], 
                                                         [('p', pp , 0, 0, 'provided'), ('h', hh , 0, 0, 'provided')], 
                                                         [(self.integrationPoints, 'linear')], 
                                                         method='raw')
        return np.real(np.asarray(zplot1))

        
    def computeCovMatrices(self, aCartPointingCoords, aCartNGSCoords, xp=np):
        points = aCartPointingCoords.shape[0]
        scaleF = (500.0/(2*np.pi))**2
        matCaaValue = xp.zeros((2,2), dtype=xp.float32)
        matCasValue = xp.zeros((2*points,6), dtype=xp.float32)
        matCssValue = xp.zeros((6,6), dtype=xp.float32)
        matCaaValue[0,0] = self.covValue(2, 2, xp.asarray([1e-10, 1e-10]), xp.asarray([self.Cn2RefHeight]))[0,0]
        matCaaValue[1,1] = self.covValue(3, 3, xp.asarray([1e-10, 1e-10]), xp.asarray([self.Cn2RefHeight]))[0,0]
        hh = xp.asarray(self.Cn2Heights)
        inputsArray = np.zeros( 3*points + 9, dtype=complex)
        iidd = 0
        for kk in [0,1,2]:
            vv = np.ones((points,2))
            vv[:,0] *= aCartNGSCoords[kk,0]
            vv[:,1] *= aCartNGSCoords[kk,1]
            polarPointingCoordsD = cartesianToPolar2(aCartPointingCoords- vv)
            polarPointingCoordsD[:,1] *= degToRad
            polarPointingCoordsD[:,0] *= arcsecsToRadians
            polarPointingCoordsD[:,0] = np.maximum( polarPointingCoordsD[:,0], 1e-9*np.ones(points))
            pp = polarPointingCoordsD[:,0]*xp.exp(1j*polarPointingCoordsD[:,1])
            inputsArray[points*iidd:points*(iidd+1)] = pp
            iidd = iidd+1
        iidd=0
        for kk1 in [0,1,2]:
            for kk2 in [0,1,2]:
                polarPointingCoordsD = cartesianToPolar(aCartNGSCoords[kk1,:]-aCartNGSCoords[kk2,:])
                polarPointingCoordsD[1] *= degToRad
                polarPointingCoordsD[0] *= arcsecsToRadians
                polarPointingCoordsD[0] = max( polarPointingCoordsD[0], 1e-9)
                pp = polarPointingCoordsD[0]*xp.exp(1j*polarPointingCoordsD[1])
                inputsArray[3*points+iidd] = pp
                iidd = iidd+1
        _idx0 = {2:[0,2,4], 3:[1,3,5]}
        for ii in [2,3]:
            for jj in [2,3]:
                outputArray1 = self.covValue(ii, jj, inputsArray, hh)
                for pind in range(points):
                    for hidx, h_weight in enumerate(self.Cn2Weights):
                        matCasValue[ii-2+pind*2][_idx0[jj]] +=  h_weight*outputArray1[pind:3*points:points, hidx]
                        if pind==0:
                            matCssValue[ xp.ix_(_idx0[ii], _idx0[jj]) ] +=  xp.reshape( h_weight*outputArray1[3*points:,hidx], (3,3))
        return scaleF*matCaaValue, scaleF*matCasValue, scaleF*matCssValue

    
    def loadWindPsd(self, filename):
        hdul = fits.open(filename)
        psd_data = np.asarray(hdul[0].data, np.float32)
        hdul.close()
        psd_freq = np.asarray(np.linspace(0.5, 250.0, 500))
        psd_tip_wind = np.zeros((500))
        psd_tilt_wind = np.zeros((500))
        psd_tip_wind[0:200] = psd_data[1,:]
        psd_tilt_wind[0:200] = psd_data[2,:]
        return psd_freq, psd_tip_wind, psd_tilt_wind

        
    def multiCMatAssemble(self, aCartPointingCoordsV, aaCartNGSCoords, aCnn, aC1):
        xp = np
        points = aCartPointingCoordsV.shape[0]
        Ctot = np.zeros((2*points,2))
        R, RT = self.buildReconstuctor2(aCartPointingCoordsV, aaCartNGSCoords)
        Caa, Cas, Css = self.computeCovMatrices(xp.asarray(aCartPointingCoordsV), xp.asarray(aaCartNGSCoords), xp=np)
        for i in range(points):
            Ri = R[2*i:2*(i+1),:]
            RTi = RT[:, 2*i:2*(i+1)]
            Casi = Cas[2*i:2*(i+1),:]
            C2b = xp.dot(Ri, xp.dot(Css, RTi)) - xp.dot(Casi, RTi) - xp.dot(Ri, Casi.transpose())
            C3 = xp.dot(Ri, xp.dot(xp.asarray(aCnn), RTi))
            ss = xp.asarray(aC1) + Caa + C2b + C3
            Ctot[2*i:2*(i+1),:] = ss
        return Ctot

        
    def CMatAssemble(self, aCartPointingCoordsV, aaCartNGSCoords, aCnn, aC1):
        R, RT = self.buildReconstuctor2(np.asarray(aCartPointingCoordsV), aaCartNGSCoords)
        Caa, Cas, Css = self.computeCovMatrices(np.asarray(aCartPointingCoordsV), aaCartNGSCoords)        
        C2 = Caa + np.dot(R, np.dot(Css, RT)) - np.dot(Cas, RT) - np.dot(R, Cas.transpose())
        C3 = np.dot(R, np.dot(aCnn, RT))
        Ctot = aC1 + C2 + C3
        return Ctot

        
    def computeTotalResidualMatrix(self, aCartPointingCoords, aCartNGSCoords, aNGS_flux, aNGS_SR_1650, aNGS_FWHM_mas):
        nPointings = aCartPointingCoords.shape[0]
        C1 = np.zeros((2,2))
        Cnn = np.zeros((6,6))
        maxFluxIndex = np.where(aNGS_flux==np.amax(aNGS_flux))
        for starIndex in [0,1,2]:
            bias, amu, avar = self.computeBias(aNGS_flux[starIndex], aNGS_SR_1650[starIndex], aNGS_FWHM_mas[starIndex]) # one scalar, two tuples of 2
            var1x = avar[0] * self.pixel_scale_LO**2
            nr = self.computeNoiseResidual(0.25, 250.0, 1000, var1x, bias, gpulib )
            wr = self.computeWindResidual(self.psd_freq, self.psd_tip_wind, self.psd_tilt_wind, var1x, bias, gpulib )
            Cnn[2*starIndex,2*starIndex] = nr[0]
            Cnn[2*starIndex+1,2*starIndex+1] = nr[1]
            if starIndex == maxFluxIndex[0][0]:
                C1[0,0] = wr[0]
                C1[1,1] = wr[1]
        # C1 and Cnn do not depend on aCartPointingCoords[i]
        Ctot = self.multiCMatAssemble(aCartPointingCoords, aCartNGSCoords, Cnn, C1)
        return Ctot.reshape((nPointings,2,2))


    def ellipsesFromCovMats(self, Ctot):
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
            scale = (np.pi/(180*3600*1000) * self.TelescopeDiameter / (4*1e-9))        
            return th, np.sqrt(smax)/scale, np.sqrt(smin)/scale

        def computeCovEllispses(Ctot):
            nPointings = Ctot.shape[0]
            result = np.zeros((nPointings, 3))
            covEllipses = np.zeros(nPointings)
            for i in range(nPointings):
                result[i] = computeCovEllispse(Ctot[i])
            return result

        return computeCovEllispses(Ctot)

    
#        if not mono and nPointings>1:
#            # while C2 and C3 do        
#            inputs = aCartPointingCoords.tolist()
#            pool_size = int( min( mp.cpu_count()/2, nPointings) )
#            semaphore = mp.Semaphore()
#            pool = mp.Pool(initializer=self.initializer, initargs=[self, semaphore], processes=pool_size)
#            pool_outputs = pool.map(functools.partial(self.CMatAssemble, aaCartNGSCoords=aCartNGSCoords, aCnn=Cnn, aC1=C1) , inputs)
#            pool.close()
#            pool.join()
#            return pool_outputs
#        else:

#    def initializer(self, semaphore):
#        """This function is run at the Pool startup. 
#        Use it to set your Semaphore object in the child process.#
#
#        """
#        self.mutex = semaphore
