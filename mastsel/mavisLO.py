from mastsel.mavisUtilities import *
from mastsel.mavisFormulas import *
import functools
import multiprocessing as mp
from configparser import ConfigParser
import os

class MavisLO(object):

    def __init__(self, path, parametersFile, verbose=False):

        self.verbose=verbose

        parser = ConfigParser()
        parser.read( os.path.join(path, parametersFile + '.ini') )
        #
        # SETTING PARAMETERS READ FROM FILE       
        #
        self.TelescopeDiameter      = eval(parser.get('telescope', 'TelescopeDiameter'))
        self.ZenithAngle            = eval(parser.get('telescope', 'ZenithAngle'))
        self.TechnicalFoV           = eval(parser.get('telescope', 'TechnicalFoV'))
        self.ObscurationRatio       = eval(parser.get('telescope', 'ObscurationRatio'))
        if parser.has_option('telescope', 'windPsdFile'):
            windPsdFile = eval(parser.get('telescope', 'windPsdFile'))
            self.psd_freq, self.psd_tip_wind, self.psd_tilt_wind = self.loadWindPsd(windPsdFile)
        else:
            print('No windPsdFile file is set.')
            self.psd_freq = np.asarray(np.linspace(0.5, 250.0, 500))
            self.psd_tip_wind = np.zeros((500))
            self.psd_tilt_wind = np.zeros((500))
            
        self.AtmosphereWavelength   = eval(parser.get('atmosphere', 'Wavelength'))
        self.L0                     = eval(parser.get('atmosphere', 'L0'))
        self.Cn2Weights             = eval(parser.get('atmosphere', 'Cn2Weights'))
        self.Cn2Heights             = eval(parser.get('atmosphere', 'Cn2Heights'))
        
        SensingWavelength_LO = eval(parser.get('sources_LO', 'Wavelength'))
        if isinstance(SensingWavelength_LO, list):
            self.SensingWavelength_LO = SensingWavelength_LO[0]
        else:
            self.SensingWavelength_LO = SensingWavelength_LO

        self.NumberLenslets         = eval(parser.get('sensor_LO', 'NumberLenslets'))

        self.N_sa_tot_LO            = self.NumberLenslets[0]**2
        if self.NumberLenslets[0] > 2:
            self.N_sa_tot_LO        = int ( np.floor( self.N_sa_tot_LO * np.pi/4.0 * (1.0 - self.ObscurationRatio**2) ) )
            
        self.PixelScale_LO          = eval(parser.get('sensor_LO', 'PixelScale'))
        self.WindowRadiusWCoG_LO    = eval(parser.get('sensor_LO', 'WindowRadiusWCoG'))
        self.sigmaRON_LO            = eval(parser.get('sensor_LO', 'SigmaRON'))
        if self.sigmaRON_LO == 0:
            self.sigmaRON_LO = 1e-6
        self.ExcessNoiseFactor_LO   = eval(parser.get('sensor_LO', 'ExcessNoiseFactor'))
        self.Dark_LO                = eval(parser.get('sensor_LO', 'Dark'))
        self.skyBackground_LO       = eval(parser.get('sensor_LO', 'SkyBackground'))
        self.ThresholdWCoG_LO       = eval(parser.get('sensor_LO', 'ThresholdWCoG'))
        self.NewValueThrPix_LO      = eval(parser.get('sensor_LO', 'NewValueThrPix'))
        
        if parser.has_option('sensor_LO', 'noNoise'):
            self.noNoise = eval(parser.get('sensor_LO', 'noNoise'))
        else:
            self.noNoise = False
            
        self.DmHeights              = eval(parser.get('DM', 'DmHeights'))
        
        self.SensorFrameRate_LO     = eval(parser.get('RTC', 'SensorFrameRate_LO'))
        self.loopDelaySteps_LO      = eval(parser.get('RTC', 'LoopDelaySteps_LO'))
        
        defaultCompute = 'GPU'
        defaultIntegralDiscretization1 = 1000
        defaultIntegralDiscretization2 = 4000
        self.computationPlatform    = eval(parser.get('COMPUTATION', 'platform', fallback='defaultCompute'))
        self.integralDiscretization1 = eval(parser.get('COMPUTATION', 'integralDiscretization1', fallback='defaultIntegralDiscretization1'))
        self.integralDiscretization2 = eval(parser.get('COMPUTATION', 'integralDiscretization2', fallback='defaultIntegralDiscretization2'))
        
        if parser.has_option('atmosphere', 'r0_Value') and parser.has_option('atmosphere', 'Seeing'):
            print('%%%%%%%% ATTENTION %%%%%%%%')
            print('You must provide r0_Value or Seeing value, not both, ')
            print('Seeing parameter will be used, r0_Value will be discarded!\n')
        
        if parser.has_option('atmosphere', 'Seeing'):
            self.Seeing = eval(parser.get('atmosphere', 'Seeing'))
            self.r0_Value = 0.976*self.AtmosphereWavelength/self.Seeing*206264.8 # old: 0.15
        else:
            self.r0_Value = eval(parser.get('atmosphere', 'r0_Value'))
            
        testWindspeedIsValid = False
        if parser.has_option('atmosphere', 'testWindspeed'):
            testWindspeed = parser.get('atmosphere', 'testWindspeed')
            try:
                testWindspeed = float(testWindspeed)
                testWindspeedIsValid = True
            except:
                testWindspeedIsValid = False
            
        if parser.has_option('atmosphere', 'WindSpeed') and parser.has_option('atmosphere', 'testWindspeed'):
            if testWindspeedIsValid:
                print('%%%%%%%% ATTENTION %%%%%%%%')
                print('You must provide WindSpeed or testWindspeed value, not both, ')
                print('testWindspeed parameter will be used, WindSpeed will be discarded!\n')
            
        if testWindspeedIsValid:
            self.WindSpeed = eval(parser.get('atmosphere', 'testWindspeed'))
        else:
            self.wSpeed = eval(parser.get('atmosphere', 'WindSpeed'))
            self.WindSpeed = (np.dot( np.power(np.asarray(self.wSpeed), 5.0/3.0), np.asarray(self.Cn2Weights) ) / np.sum( np.asarray(self.Cn2Weights) ) ) ** (3.0/5.0)   
        #
        # END OF SETTING PARAMETERS READ FROM FILE       
        #
        
        airmass = 1/np.cos(self.ZenithAngle*np.pi/180)
        self.r0_Value = self.r0_Value * airmass**(-3.0/5.0)
                 
#        self.mutex = None
        self.imax = 30
        self.zmin = 0.03
        self.zmax = 30
        self.integrationPoints = self.integralDiscretization1
        self.psdIntegrationPoints = self.integralDiscretization2
        self.largeGridSize = 200
        self.downsample_factor = 4
        self.smallGridSize = 2*self.WindowRadiusWCoG_LO
        self.p_offset = 1.0 # 1/4 pixel on medium grid
        self.mediumGridSize = int(self.largeGridSize/self.downsample_factor)
        self.mediumShape = (self.mediumGridSize,self.mediumGridSize)
        self.mediumPixelScale = self.PixelScale_LO/self.downsample_factor
        self.zernikeCov_rh1 = MavisFormulas.getFormulaRhs('ZernikeCovarianceD')
        self.zernikeCov_lh1 = MavisFormulas.getFormulaLhs('ZernikeCovarianceD')
        self.sTurbPSDTip, self.sTurbPSDTilt = self.specializedTurbFuncs()
        self.fCValue = self.specializedC_coefficient()
        self.fTipS_LO, self.fTiltS_LO = self.specializedNoiseFuncs()
        self.fTipS, self.fTiltS = self.specializedWindFuncs()
        self.specializedCovExprs = self.buildSpecializedCovFunctions()
        self.aFunctionM, self.expr0M = self.specializedMeanVarFormulas('truncatedMeanComponents')
        self.aFunctionV, self.expr0V = self.specializedMeanVarFormulas('truncatedVarianceComponents')
        self.aFunctionMGauss = self.specializedGMeanVarFormulas('GaussianMean')
        self.aFunctionVGauss = self.specializedGMeanVarFormulas('GaussianVariance')

        if self.computationPlatform=='GPU':
            self.mIt = Integrator(cp, cp.float64, '')
            self.mItcomplex = Integrator(cp, cp.complex64, '')
            self.platformlib = gpulib
        else:
            self.mIt = Integrator(np, np.float, '')
            self.mItcomplex = Integrator(np, np.complex, '')
            self.platformlib = cpulib

        

    # specialized formulas, mostly substituting parameter with mavisParametrs.py values
    def specializedIM(self, alib=cpulib):
        apIM = mf['interactionMatrixNGS']
        apIM = subsParamsByName(apIM, {'D':self.TelescopeDiameter, 'r_FoV':self.TechnicalFoV*arcsecsToRadians/2.0, 'H_DM':max(self.DmHeights)})
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

    def specializedGMeanVarFormulas(self, kind):
        dd0 = {'t':self.ThresholdWCoG_LO, 'nu':self.NewValueThrPix_LO, 'sigma_RON':self.sigmaRON_LO}
        dd1 = {'b':self.Dark_LO/self.SensorFrameRate_LO}
        dd2 = {'F':self.ExcessNoiseFactor_LO}
        expr0 = mf[kind]
        expr0 = subsParamsByName( expr0, {**dd0, **dd1, **dd2} )
        return expr0

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
        self.fTipS_LO1 = subsParamsByName(mf['completeIntegralTipLO'], dict1 ).function
        self.fTiltS_LO1 = subsParamsByName(mf['completeIntegralTiltLO'], dict1).function
#        self.fTipS_LO1 = sp.simplify(subsParamsByName(mf['completeIntegralTipLO'], dict1 ).function)
#        self.fTiltS_LO1 = sp.simplify(subsParamsByName(mf['completeIntegralTiltLO'], dict1).function)
        return self.fTipS_LO1, self.fTiltS_LO1

    
    def specializedWindFuncs(self):
        dict1 = {'d':self.loopDelaySteps_LO, 'f_loop':self.SensorFrameRate_LO}
        self.fTipS1 = subsParamsByName(mf['completeIntegralTip'], dict1).function
        self.fTiltS1 = subsParamsByName(mf['completeIntegralTilt'], dict1).function
#        self.fTipS1 = sp.simplify(subsParamsByName(mf['completeIntegralTip'], dict1).function)
#        self.fTiltS1 = sp.simplify(subsParamsByName(mf['completeIntegralTilt'], dict1).function)
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
        nstars = aCartNGSCoords.shape[0]
        if nstars==1:
            R_1 = np.zeros((2*npointings, 2*nstars))
            for k in range(npointings):
                R_1[2*k:2*(k+1), :] = np.identity(2)

            return R_1, R_1.transpose()
        else:        
            P, P_func = self.specializedIM()
            p_mat_list = []
            for ii in range(nstars):
                p_mat_list.append(P_func(aCartNGSCoords[ii,0]*arcsecsToRadians, aCartNGSCoords[ii,1]*arcsecsToRadians))
            P_mat = np.vstack(p_mat_list) # aka Interaction Matrix, im

    #        rec_tomo = scipy.linalg.pinv(P_mat) # aka W, 5x6    
            u, s, vh = np.linalg.svd(P_mat)
            sv_threshold = 0.05 * s[0]
            s_inv = np.reciprocal(s)
            sRes = np.where(s < sv_threshold, 0, s_inv)

            sRes = np.diag(sRes)
            sRes = np.append(sRes, np.zeros((1,sRes.shape[1])), axis=0 )
            if nstars==3:
                sRes = sRes.transpose()

            rec_tomo = vh.transpose() @  ( sRes  @ u.transpose() )

            vx = np.asarray(aCartPointingCoordsV[:,0])
            vy = np.asarray(aCartPointingCoordsV[:,1])
            R_1 = np.zeros((2*npointings, 2*nstars))
            for k in range(npointings):
                P_alpha1 = P_func(vx[k]*arcsecsToRadians, vy[k]*arcsecsToRadians)
                R_1[2*k:2*(k+1), :] = cp.dot(P_alpha1, rec_tomo)

            return R_1, R_1.transpose()
    
    def compute2DMeanVar(self, aFunction, expr0, gaussianPointsM, expr1):
        gaussianPoints = gaussianPointsM.reshape(self.smallGridSize*self.smallGridSize)
        aIntegral = sp.Integral(aFunction, (getSymbolByName(aFunction, 'z'), self.zmin, self.zmax), (getSymbolByName(aFunction, 'i'), 1, int(self.imax)) )
        paramsAndRanges = [( 'f_k', gaussianPoints, 0.0, 0.0, 'provided' )]
        lh = sp.Function('B')(getSymbolByName(aFunction, 'f_k'))
        xplot1, zplot1 = self.mIt.IntegralEval(lh, aIntegral, paramsAndRanges, [ (self.integrationPoints//2, 'linear'), (self.imax, 'linear')], 'raw')
        ssx, s0 = self.mIt.functionEval(expr0, paramsAndRanges )
        zplot1 = zplot1 + s0
        lh = sp.Function('B')(getSymbolByName(expr1, 'f_k'))
        ssx, zplot2 = self.mIt.functionEval(expr1, paramsAndRanges )
        rr = np.where(gaussianPoints + self.Dark_LO/self.SensorFrameRate_LO < 10.0, zplot1, zplot2)
        rr = rr.reshape((self.smallGridSize,self.smallGridSize))
        return ssx, rr

    
    def meanVarSigma(self, gaussianPoints):
        xplot1, mu_ktr_array = self.compute2DMeanVar( self.aFunctionM, self.expr0M, gaussianPoints, self.aFunctionMGauss)
        xplot2, var_ktr_array = self.compute2DMeanVar( self.aFunctionV, self.expr0V, gaussianPoints, self.aFunctionVGauss)
        var_ktr_array = var_ktr_array - mu_ktr_array**2
        sigma_ktr_array = np.sqrt(var_ktr_array.astype(np.float32))
        return mu_ktr_array, var_ktr_array, sigma_ktr_array

        
    def computeBias(self, aNGS_flux, aNGS_SR_LO, aNGS_FWHM_mas):
        gridSpanArcsec= self.mediumPixelScale*self.largeGridSize/1000
        gridSpanRad = gridSpanArcsec/radiansToArcsecs
        
        diffNGS_FWHM_mas = self.SensingWavelength_LO/(self.TelescopeDiameter)*radiansToArcsecs*1000
        
        peakValue = aNGS_flux/self.SensorFrameRate_LO*aNGS_SR_LO*4.0*np.log(2)/(np.pi*(diffNGS_FWHM_mas*2.0/self.mediumPixelScale)**2)
        if self.verbose:
            print('mavisLO.computeBias, peakValue',peakValue)
        peakValueNoFlux = aNGS_SR_LO*4.0*np.log(2)/(np.pi*(diffNGS_FWHM_mas*2.0/self.mediumPixelScale)**2)

#        peakValue = peakValue/(40**self.TelescopeDiameter/2)
#        peakValueNoFlux = aNGS_SR_LO*4.0*np.log(2)/(np.pi*(self.SensingWavelength_LO/(self.TelescopeDiameter/2)*radiansToArcsecs*1000/self.mediumPixelScale)**2)

        xCoords=np.asarray(np.linspace(-self.largeGridSize/2.0+0.5, self.largeGridSize/2.0-0.5, self.largeGridSize), dtype=np.float32)
        yCoords=np.asarray(np.linspace(-self.largeGridSize/2.0+0.5, self.largeGridSize/2.0-0.5, self.largeGridSize), dtype=np.float32)
        xGrid, yGrid = np.meshgrid( xCoords, yCoords, sparse=False, copy=True)
        
        FWHM_coeff = np.sqrt(np.abs(aNGS_FWHM_mas**2 - diffNGS_FWHM_mas**2 ))
        
        subapNGS_FWHM_mas = self.SensingWavelength_LO/(self.TelescopeDiameter/self.NumberLenslets[0])*radiansToArcsecs*1000
        
        aNGS_FWHM_mas_mod = np.sqrt( FWHM_coeff**2 + subapNGS_FWHM_mas**2 )
        asigma = aNGS_FWHM_mas_mod/sigmaToFWHM/self.mediumPixelScale
        g2d = peakValue * simple2Dgaussian( xGrid, yGrid, 0, 0, asigma)

#        asigma = aNGS_FWHM_mas/sigmaToFWHM/self.mediumPixelScale
#        asigma *= 40.0
#        g2d = peakValue * simple2Dgaussian( xGrid, yGrid, 0, 0, asigma)

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

        if self.computationPlatform=='GPU':
            xplot1, zplot1 = self.mIt.IntegralEvalE(self.sTurbPSDTip, [paramAndRange], [(self.psdIntegrationPoints, 'linear')], 'rect')
        else:
            pool_size = int( min( mp.cpu_count(), freq_samples) )
            pool = mp.Pool(processes=pool_size)
            fx = np.linspace(fmin, fmax, freq_samples)
            paramsAndRangesG = [( 'f', fxx, 0.0, 0.0, 'provided' ) for fxx in fx]
            pool_outputs = pool.map(functools.partial(self.mIt.IntegralEvalE, eq=self.sTurbPSDTip, integrationVarsSamplingSchemes=[(self.psdIntegrationPoints, 'linear')], method='rect') , [paramAndRange])
            pool.close()
            pool.join()
            for rr in pool_outputs:
                xplot1.append(rr[0])
                zplot1.append(rr[1])

        #print('x,z:', len(xplot1), len(zplot1))
        psd_freq = xplot1[0]
        psd_tip_wind = zplot1*scaleFactor
        xplot1, zplot1 = self.mIt.IntegralEvalE(self.sTurbPSDTilt, [paramAndRange], [(self.psdIntegrationPoints, 'linear')], 'rect')
        psd_tilt_wind = zplot1*scaleFactor
        return psd_tip_wind, psd_tilt_wind

        
    def computeNoiseResidual(self, fmin, fmax, freq_samples, varX, bias, alib):
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

        
    def computeWindResidual(self, psd_freq, psd_tip_wind0, psd_tilt_wind0, var1x, bias, alib):
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
        xplot1, zplot1 = self.mItcomplex.IntegralEval(sp.Function('C_v')(p, h),
                                                         self.specializedCovExprs[ii+10*jj], 
                                                         [('p', pp , 0, 0, 'provided'), ('h', hh , 0, 0, 'provided')], 
                                                         [(self.integrationPoints, 'linear')], 
                                                         method='raw')
        return np.real(np.asarray(zplot1))

        
    def computeCovMatrices(self, aCartPointingCoords, aCartNGSCoords, xp=np):
        points = aCartPointingCoords.shape[0]
        nstars = aCartNGSCoords.shape[0]        
        scaleF = (500.0/(2*np.pi))**2
        matCaaValue = xp.zeros((2,2), dtype=xp.float32)
        matCasValue = xp.zeros((2*points,2*nstars), dtype=xp.float32)
        matCssValue = xp.zeros((2*nstars,2*nstars), dtype=xp.float32)
        matCaaValue[0,0] = self.covValue(2, 2, xp.asarray([1e-10, 1e-10]), xp.asarray([1]))[0,0]
        matCaaValue[1,1] = self.covValue(3, 3, xp.asarray([1e-10, 1e-10]), xp.asarray([1]))[0,0]
        hh = xp.asarray(self.Cn2Heights)
        inputsArray = np.zeros( nstars*points + nstars*nstars, dtype=complex)
        iidd = 0
        for kk in range(nstars):
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
        for kk1 in range(nstars):
            for kk2 in range(nstars):
                polarPointingCoordsD = cartesianToPolar(aCartNGSCoords[kk1,:]-aCartNGSCoords[kk2,:])
                polarPointingCoordsD[1] *= degToRad
                polarPointingCoordsD[0] *= arcsecsToRadians
                polarPointingCoordsD[0] = max( polarPointingCoordsD[0], 1e-9)
                pp = polarPointingCoordsD[0]*xp.exp(1j*polarPointingCoordsD[1])
                inputsArray[nstars*points+iidd] = pp
                iidd = iidd+1
        
        _idx0 = {2:[0], 3:[1]}
        if nstars==3:
            _idx0 = {2:[0,2,4], 3:[1,3,5]}
        elif nstars==2:
            _idx0 = {2:[0,2], 3:[1,3]}

        for ii in [2,3]:
            for jj in [2,3]:
                outputArray1 = self.covValue(ii, jj, inputsArray, hh)
                for pind in range(points):
                    for hidx, h_weight in enumerate(self.Cn2Weights):
                        matCasValue[ii-2+pind*2][_idx0[jj]] +=  h_weight*outputArray1[pind:nstars*points:points, hidx]
                        if pind==0:
                            matCssValue[ xp.ix_(_idx0[ii], _idx0[jj]) ] +=  xp.reshape( h_weight*outputArray1[nstars*points:,hidx], (nstars,nstars))
        return scaleF*matCaaValue, scaleF*matCasValue, scaleF*matCssValue

    
    def loadWindPsd(self, filename):
        hdul = fits.open(filename)
        psd_data = np.asarray(hdul[0].data, np.float32)
        hdul.close()
        psd_freq = np.asarray(np.linspace(0.5, 250.0, 500))
        psd_tip_wind = np.zeros((500))
        psd_tilt_wind = np.zeros((500))
        psd_tip_wind[0:200] = psd_data[1,:] #TODO here we must make an interpolation using the frequencies defined in the filename
        psd_tilt_wind[0:200] = psd_data[2,:] #TODO same as above
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
            # sum tomography (Caa + C2b), noise (C3), wind (aC1) errors
            if self.noNoise:
                ss = xp.asarray(aC1) + Caa + C2b 
                print('WARNING: LO noise is not active!')
            else:
                ss = xp.asarray(aC1) + Caa + C2b + C3
            Ctot[2*i:2*(i+1),:] = ss
        return Ctot

        
    def CMatAssemble(self, aCartPointingCoordsV, aaCartNGSCoords, aCnn, aC1):
        R, RT = self.buildReconstuctor2(np.asarray(aCartPointingCoordsV), aaCartNGSCoords)
        Caa, Cas, Css = self.computeCovMatrices(np.asarray(aCartPointingCoordsV), aaCartNGSCoords)        
        C2 = Caa + np.dot(R, np.dot(Css, RT)) - np.dot(Cas, RT) - np.dot(R, Cas.transpose())
        C3 = np.dot(R, np.dot(aCnn, RT))
        # sum tomography (C2), noise (C3), wind (aC1) errors
        if self.noNoise:
            ss = aC1 + C2
            print('WARNING: LO noise is not active!')
        else:
            Ctot = aC1 + C2 + C3
        return Ctot

        
    def computeTotalResidualMatrix(self, aCartPointingCoords, aCartNGSCoords, aNGS_flux, aNGS_SR_LO, aNGS_FWHM_mas):
        nPointings = aCartPointingCoords.shape[0]
        maxFluxIndex = np.where(aNGS_flux==np.amax(aNGS_flux))
        nNaturalGS = aCartNGSCoords.shape[0]
        C1 = np.zeros((2,2))
        Cnn = np.zeros((2*nNaturalGS,2*nNaturalGS))


        if self.verbose:
            print('mavisLO.computeTotalResidualMatrix')
            print('             aNGS_flux',aNGS_flux)
            print('             self.N_sa_tot_LO',self.N_sa_tot_LO)
        for starIndex in range(nNaturalGS):
            bias, amu, avar = self.computeBias(aNGS_flux[starIndex], aNGS_SR_LO[starIndex], aNGS_FWHM_mas[starIndex]) # one scalar, two tuples of 2
            if self.verbose:
                print('             bias',bias)
                print('             amu',amu)
                print('             avar',avar)

            var1x = avar[0] * self.PixelScale_LO**2
            nr = self.computeNoiseResidual(0.25, 250.0, 1000, var1x, bias, self.platformlib )
            wr = self.computeWindResidual(self.psd_freq, self.psd_tip_wind, self.psd_tilt_wind, var1x, bias, self.platformlib )
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
            if np.abs(CC[1,0]) < 1e-20:
                CC[1,0] = 1e-20
            if np.abs(CC[0,1]) < 1e-20:
                CC[0,1] = 1e-20
            
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
