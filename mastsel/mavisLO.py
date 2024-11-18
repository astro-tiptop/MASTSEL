import numpy as np

from . import gpuEnabled

if not gpuEnabled:
    cp = np
else:
    import cupy as cp

from mastsel.mavisUtilities import *
from mastsel.mavisFormulas import *
from mastsel.mavisFormulas import _mavisFormulas

from sympy.physics.control.lti import TransferFunction
import functools
import multiprocessing as mp
from configparser import ConfigParser
import yaml
import os

def cpuArray(v):
    if isinstance(v,np.ndarray) or isinstance(v,np.float64) or isinstance(v, float):
        return v
    else:
        return v.get()

class MavisLO(object):
    
    def check_section_key(self, primary):
        if self.configType == 'ini':
            return self.config.has_section(primary)
        elif self.configType == 'yml':
            return primary in self.my_yaml_dict.keys()
    
    def check_config_key(self, primary, secondary):
        if self.configType == 'ini':
            return self.config.has_option(primary, secondary)
        elif self.configType == 'yml':
            if primary in self.my_yaml_dict.keys():
                return secondary in self.my_yaml_dict[primary].keys()
            else:
                return False

    def get_config_value(self, primary, secondary):
        if self.configType == 'ini':
            return eval(self.config[primary][secondary])
        elif self.configType == 'yml':
            return self.my_yaml_dict[primary][secondary]

    def __init__(self, path, parametersFile, verbose=False):

        self.verbose = verbose
        self.plot4debug = False
        self.displayEquation = False

        if self.verbose: np.set_printoptions(precision=3)
        
        filename_ini = os.path.join(path, parametersFile + '.ini')
        filename_yml = os.path.join(path, parametersFile + '.yml')

        self.error = False
        if os.path.exists(filename_yml):
            self.configType = 'yml'
            with open(filename_yml) as f:
                self.my_yaml_dict = yaml.safe_load(f)
        elif os.path.exists(filename_ini):
            self.configType = 'ini'
            self.config = ConfigParser()
            self.config.optionxform = str
            self.config.read(filename_ini)
        else:
            print('%%%%%%%% ERROR %%%%%%%%')
            print('The .ini or .yml file does not exist\n')
            self.error = True
            return
        
        self.TelescopeDiameter      = self.get_config_value('telescope','TelescopeDiameter')  
        self.ZenithAngle            = self.get_config_value('telescope','ZenithAngle')  
        self.TechnicalFoV           = self.get_config_value('telescope','TechnicalFoV')  
        self.ObscurationRatio       = self.get_config_value('telescope','ObscurationRatio')   

        self.AtmosphereWavelength   = self.get_config_value('atmosphere','Wavelength')
        self.L0                     = self.get_config_value('atmosphere','L0')
        self.Cn2Weights             = self.get_config_value('atmosphere','Cn2Weights')
        self.Cn2Heights             = self.get_config_value('atmosphere','Cn2Heights')
       
        if np.min(self.Cn2Heights) == 0:
            self.Cn2Heights[np.argmin(self.Cn2Heights)] = 1e-6

        self.Cn2HeightsMean = (np.dot( np.power(np.asarray(self.Cn2Heights), 5.0/3.0), np.asarray(self.Cn2Weights) ) / np.sum( np.asarray(self.Cn2Weights) ) ) ** (3.0/5.0)

        SensingWavelength_LO = self.get_config_value('sources_LO','Wavelength')
        if isinstance(SensingWavelength_LO, list):
            self.SensingWavelength_LO = SensingWavelength_LO[0]
        else:
            self.SensingWavelength_LO = SensingWavelength_LO

        self.NumberLenslets         = self.get_config_value('sensor_LO','NumberLenslets')

        self.N_sa_tot_LO            = self.NumberLenslets[0]**2
        if self.NumberLenslets[0] > 2:
            self.N_sa_tot_LO        = int ( np.floor( self.N_sa_tot_LO * np.pi/4.0 * (1.0 - self.ObscurationRatio**2) ) )

        self.PixelScale_LO          = self.get_config_value('sensor_LO','PixelScale')
        self.WindowRadiusWCoG_LO    = self.get_config_value('sensor_LO','WindowRadiusWCoG')
        if self.WindowRadiusWCoG_LO=='optimize':
            self.WindowRadiusWCoG_LO = 0
        self.sigmaRON_LO            = self.get_config_value('sensor_LO','SigmaRON')
        if self.sigmaRON_LO == 0:
            self.sigmaRON_LO = 1e-6
        self.ExcessNoiseFactor_LO   = self.get_config_value('sensor_LO','ExcessNoiseFactor')
        self.Dark_LO                = self.get_config_value('sensor_LO','Dark')
        self.skyBackground_LO       = self.get_config_value('sensor_LO','SkyBackground')
        # this is called t in MAVIS AOM formulas
        self.ThresholdWCoG_LO       = self.get_config_value('sensor_LO','ThresholdWCoG')
        # this is called v (nu greek letter) in MAVIS AOM formulas
        self.NewValueThrPix_LO      = self.get_config_value('sensor_LO','NewValueThrPix')

        
        if self.check_config_key('sensor_LO','noNoise'):
            self.noNoise = self.get_config_value('sensor_LO','noNoise')
        else:
            self.noNoise = False

        self.DmHeights              = self.get_config_value('DM','DmHeights')

        if self.check_config_key('sensor_LO','filtZernikeCov'):
            self.filtZernikeCov = self.get_config_value('sensor_LO','filtZernikeCov')
            if self.filtZernikeCov and len(self.DmHeights) == 1:
                print('WARNING: [sensor_LO] filtZernikeCov cannot be used in systems with a single DM.')
                self.filtZernikeCov = False
        else:
            self.filtZernikeCov = False

        self.loopDelaySteps_LO      = self.get_config_value('RTC','LoopDelaySteps_LO')

        if self.check_config_key('RTC','LoopGain_LO'):
            self.LoopGain_LO            = self.get_config_value('RTC','LoopGain_LO')
        else:
            self.LoopGain_LO            = 'optimize'

        self.LoopGain_HO            = self.get_config_value('RTC','LoopGain_HO')
        self.SensorFrameRate_HO     = self.get_config_value('RTC','SensorFrameRate_HO')
        self.LoopDelaySteps_HO      = self.get_config_value('RTC','LoopDelaySteps_HO')

        if self.check_section_key('sensor_Focus'):
            self.WindowRadiusWCoG_Focus  = self.get_config_value('sensor_Focus','WindowRadiusWCoG')
            if self.WindowRadiusWCoG_Focus=='optimize':
                self.WindowRadiusWCoG_Focus = 0
            self.skyBackground_Focus     = self.get_config_value('sensor_Focus','SkyBackground')
            self.Dark_Focus              = self.get_config_value('sensor_Focus','Dark')
            self.PixelScale_Focus        = self.get_config_value('sensor_Focus','PixelScale')
            self.ExcessNoiseFactor_Focus = self.get_config_value('sensor_Focus','ExcessNoiseFactor')
            self.sigmaRON_Focus          = self.get_config_value('sensor_Focus','SigmaRON')
            self.NumberLenslets_Focus    = self.get_config_value('sensor_Focus','NumberLenslets')
        else:
            self.WindowRadiusWCoG_Focus  = self.WindowRadiusWCoG_LO
            self.skyBackground_Focus     = self.skyBackground_LO
            self.Dark_Focus              = self.Dark_LO
            self.PixelScale_Focus        = self.PixelScale_LO
            self.ExcessNoiseFactor_Focus = self.ExcessNoiseFactor_LO
            self.sigmaRON_Focus          = self.sigmaRON_LO
            self.NumberLenslets_Focus    = self.NumberLenslets

        if self.check_section_key('sensor_Focus'):
            SensingWavelength_Focus = self.get_config_value('sources_Focus','Wavelength')
            if isinstance(SensingWavelength_Focus, list):
                self.SensingWavelength_Focus = SensingWavelength_Focus[0]
            else:
                self.SensingWavelength_Focus = SensingWavelength_Focus
        else:
            self.SensingWavelength_Focus = self.SensingWavelength_LO

        if self.check_config_key('RTC','LoopGain_Focus'):
            self.LoopGain_Focus        = self.get_config_value('RTC','LoopGain_Focus')
        else:
            self.LoopGain_Focus        = 'optimize'

        if self.check_config_key('RTC','LoopDelaySteps_Focus'):
            self.loopDelaySteps_Focus  = self.get_config_value('RTC','LoopDelaySteps_Focus')
        else:
            self.loopDelaySteps_Focus  = self.loopDelaySteps_LO

        self.N_sa_tot_Focus            = self.NumberLenslets_Focus[0]**2
        if self.NumberLenslets_Focus[0] > 2:
            self.N_sa_tot_Focus        = int ( np.floor( self.N_sa_tot_Focus * np.pi/4.0 * (1.0 - self.ObscurationRatio**2) ) )

        defaultCompute = 'GPU'
        defaultIntegralDiscretization1 = 1000
        defaultIntegralDiscretization2 = 4000
        defaultSimpleVarianceComputation = True
        self.computationPlatform = defaultCompute
        self.integralDiscretization1 = defaultIntegralDiscretization1
        self.integralDiscretization2 = defaultIntegralDiscretization2
        self.simpleVarianceComputation = defaultSimpleVarianceComputation

        if self.check_section_key('COMPUTATION'):
            if self.check_config_key('COMPUTATION','platform'):
                self.computationPlatform    = self.get_config_value('COMPUTATION','platform')
            if self.check_config_key('COMPUTATION','integralDiscretization1'):
                self.integralDiscretization1 = self.get_config_value('COMPUTATION','integralDiscretization1')
            if self.check_config_key('COMPUTATION','integralDiscretization2'):
                self.integralDiscretization2 = self.get_config_value('COMPUTATION','integralDiscretization2')
            if self.check_config_key('COMPUTATION','simpleVarianceComputation'):
                self.simpleVarianceComputation = self.get_config_value('COMPUTATION','simpleVarianceComputation')

        if self.verbose:
            print('defaultSimpleVarianceComputation',self.simpleVarianceComputation)
        
        if self.check_config_key('atmosphere','r0_Value') and self.check_config_key('atmosphere','Seeing'):
            print('%%%%%%%% ATTENTION %%%%%%%%')
            print('You must provide r0_Value or Seeing value, not both, ')
            print('Seeing parameter will be used, r0_Value will be discarded!\n')

        if self.check_config_key('atmosphere','Seeing'):
            self.Seeing = self.get_config_value('atmosphere','Seeing')
            self.r0_Value = 0.976*self.AtmosphereWavelength/self.Seeing*206264.8 # old: 0.15
        else:
            self.r0_Value = self.get_config_value('atmosphere','r0_Value')

        testWindspeedIsValid = False
        if self.check_config_key('atmosphere','testWindspeed'):
            testWindspeed = self.get_config_value('atmosphere','testWindspeed')
            try:
                testWindspeed = float(testWindspeed)
                if testWindspeed > 0:
                    testWindspeedIsValid = True
                else:
                    testWindspeedIsValid = False
            except:
                testWindspeedIsValid = False

        if self.check_config_key('atmosphere','WindSpeed') and self.check_config_key('atmosphere','testWindspeed'):
            if testWindspeedIsValid:
                print('%%%%%%%% ATTENTION %%%%%%%%')
                print('You must provide WindSpeed or testWindspeed value, not both, ')
                print('testWindspeed parameter will be used, WindSpeed will be discarded!\n')

        if testWindspeedIsValid:
            self.WindSpeed = self.get_config_value('atmosphere','testWindspeed')
        else:
            self.wSpeed = self.get_config_value('atmosphere','WindSpeed')
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
        # this is N_W in the simplified variance formulas
        self.smallGridSize = 2*self.WindowRadiusWCoG_LO
        self.p_offset = 1.0 # 1/4 pixel on medium grid
        self.mediumGridSize = int(self.largeGridSize/self.downsample_factor)
        self.mediumShape = (self.mediumGridSize,self.mediumGridSize)
        self.mediumPixelScale = self.PixelScale_LO/self.downsample_factor

        # diffraction limited FWHM - full aperture
        #self.diffNGS_FWHM_mas = self.SensingWavelength_LO/(self.TelescopeDiameter)*radiansToArcsecs*1000
        # diffraction limited FWHM - one aperture
        self.subapNGS_FWHM_mas = self.SensingWavelength_LO/(self.TelescopeDiameter/self.NumberLenslets[0])*radiansToArcsecs*1000
        self.subapFocus_FWHM_mas = self.SensingWavelength_Focus/(self.TelescopeDiameter/self.NumberLenslets_Focus[0])*radiansToArcsecs*1000
        
        if self.computationPlatform=='GPU' and gpuEnabled:
            self.mIt = Integrator(cp, cp.float64, '')
            self.mItcomplex = Integrator(cp, cp.complex64, '')
            self.platformlib = gpulib
        else:
            self.mIt = Integrator(np, float, '')
            self.mItcomplex = Integrator(np, complex, '')
            self.platformlib = cpulib

        self.MavisFormulas = _mavisFormulas
        self.zernikeCov_rh1 = self.MavisFormulas.getFormulaRhs('ZernikeCovarianceD')
        self.zernikeCov_rh1_filt = self.MavisFormulas.getFormulaRhs('ZernikeCovarianceDfiltered')
        self.zernikeCov_lh1 = self.MavisFormulas.getFormulaLhs('ZernikeCovarianceD')
        self.sTurbPSDTip, self.sTurbPSDTilt = self.specializedTurbFuncs()
        self.sTurbPSDFocus, self.sSodiumPSDFocus = self.specializedFocusFuncs()
        self.fCValue = self.specializedC_coefficient()
        self.fCfocusValue = self.specializedCfocus_coefficient()
        self.specializedCovExprs = self.buildSpecializedCovFunctions()

        # this is not used for now, as the frequencies of the LO loop are passed as parameters when needed
        #self.SensorFrameRate_LO_array  = self.get_config_value('RTC','SensorFrameRate_LO')
        #if not isinstance(self.SensorFrameRate_LO_array, list):
        #    self.SensorFrameRate_LO_array  = [self.SensorFrameRate_LO_array] * len(self.get_config_value('sensor_LO','NumberPhotons'))


    # called each time we do the computatation specific to a star
    def configLOFreq(self, frequency):
        self.SensorFrameRate_LO = frequency
        self.maxLOtFreq = 0.5*self.SensorFrameRate_LO
        if self.check_config_key('telescope','windPsdFile'):
            windPsdFile = self.get_config_value('telescope','windPsdFile')
            self.psd_freq, self.psd_tip_wind, self.psd_tilt_wind = self.loadWindPsd(windPsdFile)
        else:
            if self.verbose:
                print('    WARNING: no windPsdFile file is set.')
            self.psd_freq = np.asarray(np.linspace(0.5, self.maxLOtFreq, int(2*self.maxLOtFreq)))
            self.psd_tip_wind = np.zeros((int(2*self.maxLOtFreq)))
            self.psd_tilt_wind = np.zeros((int(2*self.maxLOtFreq)))

        self.fTipS_LO, self.fTiltS_LO = self.specializedNoiseFuncs()
        self.fTipS, self.fTiltS = self.specializedWindFuncs()
        if not self.simpleVarianceComputation:
            self.aFunctionM, self.expr0M = self.specializedMeanVarFormulas('truncatedMeanComponents')
            self.aFunctionV, self.expr0V = self.specializedMeanVarFormulas('truncatedVarianceComponents')
            self.aFunctionMGauss = self.specializedGMeanVarFormulas('GaussianMean')
            self.aFunctionVGauss = self.specializedGMeanVarFormulas('GaussianVariance')


    def configFocusFreq(self, frequency):
        self.SensorFrameRate_Focus = frequency
        self.maxFocustFreq = 0.5*self.SensorFrameRate_Focus
        self.fFocusS_LO = self.specializedNoiseFocusFuncs()


    def loadWindPsd(self, filename):
        hdul = fits.open(filename)
        psd_data = np.asarray(hdul[0].data, np.float32)
        hdul.close()
        psd_freq = np.asarray(np.linspace(0.5, self.maxLOtFreq, int(2*self.maxLOtFreq)))
        psd_tip_wind = np.interp(psd_freq, psd_data[0,:], psd_data[1,:],left=0,right=0)
        psd_tilt_wind = np.interp(psd_freq, psd_data[0,:], psd_data[2,:],left=0,right=0)
        return psd_freq, psd_tip_wind, psd_tilt_wind


    # specialized formulas, mostly substituting parameter with mavisParametrs.py values
    def specializedIM(self, alib=cpulib):
        apIM = self.MavisFormulas['interactionMatrixNGS']
        apIM = subsParamsByName(apIM, {'D':self.TelescopeDiameter, 'r_FoV':self.TechnicalFoV*arcsecsToRadians/2.0, 'H_DM':max(self.DmHeights)})
        xx, yy = sp.symbols('x_1 y_1', real=True)
        apIM = subsParamsByName(apIM, {'x_NGS':xx, 'y_NGS':yy})
        apIM_func = sp.lambdify((xx, yy), apIM, modules=alib)
        if self.displayEquation:
            print('mavisLO.specializedIM')
            print('    apIM')
            try:
                display(apIM)
            except:
                print('    no apIM')
            print('    apIM_func')
            try:
                display(apIM_func)
            except:
                print('    no apIM_func')
        return apIM, apIM_func

    
    def specializedMeanVarFormulas(self, kind):
        dd0 = {'t':self.ThresholdWCoG_LO, 'nu':self.NewValueThrPix_LO, 'sigma_RON':self.sigmaRON_LO}
        dd1 = {'b':(self.Dark_LO+self.skyBackground_LO)/self.SensorFrameRate_LO}
        dd2 = {'F':self.ExcessNoiseFactor_LO}
        expr0, exprK, integral = self.MavisFormulas[kind+"0"], self.MavisFormulas[kind+"1"], self.MavisFormulas[kind+"2"]
        expr0 = subsParamsByName( expr0, {**dd0, **dd1} )
        exprK = subsParamsByName( exprK, {**dd1} )
        integral = subsParamsByName( integral,  {**dd0, **dd1, **dd2} )
        if self.displayEquation:
            print('mavisLO.specializedMeanVarFormulas')
            print('    expr0')
            try:
                display(expr0)
            except:
                print('    no expr0')
            print('    exprK')
            try:
                display(exprK)
            except:
                print('    no exprK')
            print('    integral')
            try:
                display(integral)
            except:
                print('    no integral')
        aFunction = exprK * integral.function
        return aFunction, expr0

    def specializedGMeanVarFormulas(self, kind):
        dd0 = {'t':self.ThresholdWCoG_LO, 'nu':self.NewValueThrPix_LO, 'sigma_RON':self.sigmaRON_LO}
        dd1 = {'b':(self.Dark_LO+self.skyBackground_LO)/self.SensorFrameRate_LO}
        dd2 = {'F':self.ExcessNoiseFactor_LO}
        expr0 = self.MavisFormulas[kind]
        expr0 = subsParamsByName( expr0, {**dd0, **dd1, **dd2} )
        if self.displayEquation:
            print('mavisLO.specializedGMeanVarFormulas')
            print('    expr0')
            try:
                display(expr0)
            except:
                print('    no expr0')
        return expr0

    def specializedTurbFuncs(self):
        aTurbPSDTip = subsParamsByName(self.MavisFormulas['turbPSDTip'], {'V':self.WindSpeed, 'R':self.TelescopeDiameter/2.0, 'r_0':self.r0_Value, 'L_0':self.L0, 'k_y_min':0.0001, 'k_y_max':100})
        aTurbPSDTilt = subsParamsByName(self.MavisFormulas['turbPSDTilt'], {'V':self.WindSpeed, 'R':self.TelescopeDiameter/2.0, 'r_0':self.r0_Value, 'L_0':self.L0, 'k_y_min':0.0001, 'k_y_max':100})
        if self.displayEquation:
            print('mavisLO.specializedTurbFuncs')
            print('    aTurbPSDTip')
            try:
                display(aTurbPSDTip)
            except:
                print('    no aTurbPSDTip')
            print('    aTurbPSDTilt')
            try:
                display(aTurbPSDTilt)
            except:
                print('    no aTurbPSDTilt')
        return aTurbPSDTip, aTurbPSDTilt

    def specializedFocusFuncs(self):
        aTurbPSDFocus = subsParamsByName(self.MavisFormulas['turbPSDFocus'], {'V':self.WindSpeed, 'R':self.TelescopeDiameter/2.0, 'r_0':self.r0_Value, 'L_0':self.L0, 'k_y_min':0.0001, 'k_y_max':100})
        aSodiumPSDFocus = subsParamsByName(self.MavisFormulas['sodiumPSDFocus'], {'R':self.TelescopeDiameter/2.0, 'ZenithAngle':self.ZenithAngle})
        if self.displayEquation:
            print('mavisLO.specializedFocusFuncs')
            print('    aTurbPSDFocus')
            try:
                display(aTurbPSDFocus)
            except:
                print('    no aTurbPSDFocus')
            print('    aSodiumPSDFocus')
            try:
                display(aSodiumPSDFocus)
            except:
                print('    no aSodiumPSDFocus')
        return aTurbPSDFocus, aSodiumPSDFocus
        
    def specializedC_coefficient(self):
        ffC = self.MavisFormulas['noisePropagationCoefficient'].rhs
        self.fCValue1 = subsParamsByName(ffC, {'D':self.TelescopeDiameter, 'N_sa_tot':self.N_sa_tot_LO })
        if self.displayEquation:
            print('mavisLO.specializedC_coefficient')
            print('    self.fCValue1')
            try:
                display(self.fCValue1)
            except:
                print('    no self.fCValue1')
        return self.fCValue1

    def specializedCfocus_coefficient(self):
        ffC = self.MavisFormulas['noisePropagationCoefficient'].rhs
        self.fCfocusValue1 = subsParamsByName(ffC, {'D':self.TelescopeDiameter, 'N_sa_tot':self.N_sa_tot_Focus })
        if self.displayEquation:
            print('mavisLO.specializedCfocus_coefficient')
            print('    self.fCfocusValue1')
            try:
                display(self.fCfocusValue1)
            except:
                print('    no self.fCfocusValue1')
        return self.fCfocusValue1
    
    def specializedNoiseFuncs(self):
        dict1 = {'d':self.loopDelaySteps_LO, 'f_loop':self.SensorFrameRate_LO}
        completeIntegralTipLOandTf = self.MavisFormulas['completeIntegralTipLOandTf']
        self.fTipS_LO1 = subsParamsByName(completeIntegralTipLOandTf[0], dict1).function
        self.fTipS_LO1tfW = completeIntegralTipLOandTf[1]
        self.fTipS_LO1tfN = completeIntegralTipLOandTf[2]
        self.fTipS_LO1ztfW = completeIntegralTipLOandTf[3]
        self.fTipS_LO1ztfN = completeIntegralTipLOandTf[4]
        completeIntegralTiltLOandTf = self.MavisFormulas['completeIntegralTiltLOandTf']
        self.fTiltS_LO1 = subsParamsByName(completeIntegralTiltLOandTf[0], dict1).function
        self.fTiltS_LO1tfW = completeIntegralTiltLOandTf[1]
        self.fTiltS_LO1tfN = completeIntegralTiltLOandTf[2]
        self.fTiltS_LO1ztfW = completeIntegralTiltLOandTf[3]
        self.fTiltS_LO1ztfN = completeIntegralTiltLOandTf[4]
#        self.fTipS_LO1 = sp.simplify(subsParamsByName(self.MavisFormulas['completeIntegralTipLO'], dict1 ).function)
#        self.fTiltS_LO1 = sp.simplify(subsParamsByName(self.MavisFormulas['completeIntegralTiltLO'], dict1).function)
        if self.displayEquation:
            print('mavisLO.specializedNoiseFuncs')
            print('    self.fTipS_LO1')
            try:
                display(self.fTipS_LO1)
            except:
                print('    no self.fTipS_LO1')
            print('    self.fTiltS_LO1')
            try:
                display(self.fTiltS_LO1)
            except:
                print('    no self.fTiltS_LO1')
        return self.fTipS_LO1, self.fTiltS_LO1


    def specializedNoiseFocusFuncs(self):
        dict1 = {'d':self.loopDelaySteps_Focus, 'f_loop':self.SensorFrameRate_Focus}
        completeIntegralTipLOandTf = self.MavisFormulas['completeIntegralTipLOandTf']
        self.fFocusS_LO1 = subsParamsByName(completeIntegralTipLOandTf[0], dict1).function
        self.fFocusS_LO1tfW = completeIntegralTipLOandTf[1]
        self.fFocusS_LO1tfN = completeIntegralTipLOandTf[2]
        self.fFocusS_LO1ztfW = completeIntegralTipLOandTf[3]
        self.fFocusS_LO1ztfN = completeIntegralTipLOandTf[4]
        if self.displayEquation:
            print('mavisLO.specializedNoiseFocusFuncs')
            print('    self.fFocusS_LO1')
            try:
                display(self.fFocusS_LO1)
            except:
                print('    no self.fFocusS_LO1')
        return self.fFocusS_LO1


    def specializedWindFuncs(self):
        dict1 = {'d':self.loopDelaySteps_LO, 'f_loop':self.SensorFrameRate_LO}
        completeIntegralTipAndTf = self.MavisFormulas['completeIntegralTipAndTf']
        self.fTipS1 = subsParamsByName(completeIntegralTipAndTf[0], dict1).function
        self.fTipS1tfW = completeIntegralTipAndTf[1]
        self.fTipS1tfN = completeIntegralTipAndTf[2]
        self.fTipS1ztfW = completeIntegralTipAndTf[3]
        self.fTipS1ztfN = completeIntegralTipAndTf[4]
        completeIntegralTiltAndTf = self.MavisFormulas['completeIntegralTiltAndTf']
        self.fTiltS1 = subsParamsByName(completeIntegralTiltAndTf[0], dict1).function
        self.fTiltS1tfW = completeIntegralTiltAndTf[1]
        self.fTiltS1tfN = completeIntegralTiltAndTf[2]
        self.fTiltS1ztfW = completeIntegralTiltAndTf[3]
        self.fTiltS1ztfN = completeIntegralTiltAndTf[4]
#        self.fTipS1 = sp.simplify(subsParamsByName(self.MavisFormulas['completeIntegralTip'], dict1).function)
#        self.fTiltS1 = sp.simplify(subsParamsByName(self.MavisFormulas['completeIntegralTilt'], dict1).function)
        if self.displayEquation:
            print('mavisLO.specializedWindFuncs')
            print('    self.fTipS1')
            try:
                display(self.fTipS1)
            except:
                print('    no self.fTipS1')
            print('    self.fTiltS1')
            try:
                display(self.fTiltS1)
            except:
                print('    no self.fTiltS1')
        return self.fTipS1, self.fTiltS1
    
    
    def buildSpecializedCovFunctions(self):
        covValue_integrationLimits = (sp.symbols('f', positive=True), 1e-3, 10.0)
        p = sp.symbols('p', real=False)
        cov_expr={}
        if self.filtZernikeCov:
            paramDictBaseCov = { 'L_0': self.L0, 'r_0': self.r0_Value, 'R_1': self.TelescopeDiameter/2.0, 'R_2': self.TelescopeDiameter/2.0, \
                                 'fr_ho': self.SensorFrameRate_HO, 'fov_radius': 0.5*self.TechnicalFoV, 'h_mean': self.Cn2HeightsMean, \
                                 'wind_speed_mean': self.WindSpeed}
            if self.displayEquation:
                print('zernikeCov_rh1_filt')
                try:
                    display(self.zernikeCov_rh1_filt)
                except:
                    print('    no zernikeCov_rh1_filt')
            for ii in [2,3,4]:
                for jj in [2,3,4]:

                    aa = subsParamsByName(cov_expr_jk(self.zernikeCov_rh1_filt, ii, jj), paramDictBaseCov)
                    aaint = sp.Integral(aa, covValue_integrationLimits)
                    aaint = subsParamsByName(aaint, {'rho': sp.Abs(p), 'theta': sp.arg(p)} )
                    cov_expr[ii+10*jj] = aaint
        else:
            paramDictBaseCov = { 'L_0': self.L0, 'r_0': self.r0_Value, 'R_1': self.TelescopeDiameter/2.0, 'R_2': self.TelescopeDiameter/2.0}
            for ii in [2,3,4]:
                for jj in [2,3,4]:
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
            
            rec_tomo = np.linalg.pinv(P_mat,rcond=0.05) # aka W, 5x(2*nstars)    

            vx = np.asarray(aCartPointingCoordsV[:,0])
            vy = np.asarray(aCartPointingCoordsV[:,1])
            R_1 = np.zeros((2*npointings, 2*nstars))
            for k in range(npointings):
                P_alpha1 = P_func(vx[k]*arcsecsToRadians, vy[k]*arcsecsToRadians)
                R_1[2*k:2*(k+1), :] = cp.dot(P_alpha1, rec_tomo)

            return R_1, R_1.transpose()
    
    def compute2DMeanVar(self, aFunction, expr0, gaussianPointsM, expr1):
        gaussianPoints = gaussianPointsM.flatten()
        aIntegral = sp.Integral(aFunction, (getSymbolByName(aFunction, 'z'), self.zmin, self.zmax), (getSymbolByName(aFunction, 'i'), 1, int(self.imax)) )
        paramsAndRanges = [( 'f_k', gaussianPoints, 0.0, 0.0, 'provided' )]
        lh = sp.Function('B')(getSymbolByName(aFunction, 'f_k'))
        xplot1, zplot1 = self.mIt.IntegralEval(lh, aIntegral, paramsAndRanges, [ (self.integrationPoints//2, 'linear'), (self.imax, 'linear')], 'raw')
        ssx, s0 = self.mIt.functionEval(expr0, paramsAndRanges )
        zplot1 = zplot1 + s0
        lh = sp.Function('B')(getSymbolByName(expr1, 'f_k'))
        ssx, zplot2 = self.mIt.functionEval(expr1, paramsAndRanges )
        # magic number 10 here is due to the fact that more than 10 photons flux can be approximated witha gaussian distribution
        rr = np.where(gaussianPoints + (self.Dark_LO+self.skyBackground_LO)/self.SensorFrameRate_LO < 10.0, zplot1, zplot2)
        rr = rr.reshape(gaussianPointsM.shape)
        return ssx, rr

    
    def meanVarSigma(self, gaussianPoints):
        xplot1, mu_ktr_array = self.compute2DMeanVar( self.aFunctionM, self.expr0M, gaussianPoints, self.aFunctionMGauss)
        xplot2, var_ktr_array = self.compute2DMeanVar( self.aFunctionV, self.expr0V, gaussianPoints, self.aFunctionVGauss)
        var_ktr_array = var_ktr_array - mu_ktr_array**2
        sigma_ktr_array = np.sqrt(var_ktr_array.astype(np.float32))
        return mu_ktr_array, var_ktr_array, sigma_ktr_array


    def simplifiedComputeBiasAndVariance(self, aNGS_flux, aNGS_freq, aNGS_EE, aNGS_FWHM_mas, doLO=True):
        if doLO:
            if self.WindowRadiusWCoG_LO == 0:
                WindowRadiusWCoG = max(int(np.round((aNGS_FWHM_mas/2)/self.PixelScale_LO)), 1)
            else:
                WindowRadiusWCoG = self.WindowRadiusWCoG_LO
            skyBackground = self.skyBackground_LO
            Dark = self.Dark_LO
            PixelScale = self.PixelScale_LO
            ExcessNoiseFactor = self.ExcessNoiseFactor_LO
            sigmaRON = self.sigmaRON_LO
            subap_FWHM_mas = self.subapNGS_FWHM_mas
        else:
            if self.WindowRadiusWCoG_Focus == 0:
                WindowRadiusWCoG = max(int(np.round((aNGS_FWHM_mas/2)/self.PixelScale_Focus)), 1)
            else:
                WindowRadiusWCoG = self.WindowRadiusWCoG_Focus
            skyBackground = self.skyBackground_Focus
            Dark = self.Dark_Focus
            PixelScale = self.PixelScale_Focus
            ExcessNoiseFactor = self.ExcessNoiseFactor_Focus
            sigmaRON = self.sigmaRON_Focus
            subap_FWHM_mas = self.subapFocus_FWHM_mas

        # aNGS_flux is provided in photons/s
        aNGS_frameflux = aNGS_flux / aNGS_freq
        sky_and_back = (skyBackground + Dark)/aNGS_freq
        N_T = aNGS_FWHM_mas/PixelScale
        N_W = 2*WindowRadiusWCoG
        N_D = subap_FWHM_mas/PixelScale
        sigma_e = np.sqrt( ExcessNoiseFactor * sky_and_back + sigmaRON**2 )
        sigma_ph_fwhm = ExcessNoiseFactor*(1.0/(2.0*np.log(2.0)*aNGS_frameflux)) * ((N_T)*((N_T**2+N_W**2)/(2*N_T**2+N_W**2))) ** 2
        sigma_ron_fwhm = (np.pi/(32.0*np.log(2.0))) * ( (sigma_e/(aNGS_frameflux)) * (N_T**2+N_W**2) ) ** 2
        # in the next lines we approximate that the encircled energy present in 2 times the FWHM
        # of the PSF can be effectively used to compute the centroid
        sigma_ph_ee = (1.0/aNGS_EE) * sigma_ph_fwhm
        sigma_ron_ee = (1.0/aNGS_EE)**2 * sigma_ron_fwhm
        sigma_tot = sigma_ph_ee + sigma_ron_ee
        varx = vary = sigma_tot
        mux = muy = 0
        bias = N_W**2/(N_W**2+N_T**2)
        return (bias,(mux,muy),(varx,vary))


    def computeBiasAndVariance(self, aNGS_flux, aNGS_freq, aNGS_EE, aNGS_FWHM_mas):
        if self.WindowRadiusWCoG_LO == 0:
            WindowRadiusWCoG_LO = max(int(np.ceil((aNGS_FWHM_mas/2)/self.PixelScale_LO)),1)
            self.smallGridSize = 2*WindowRadiusWCoG_LO
        else:
            WindowRadiusWCoG_LO = self.WindowRadiusWCoG_LO
        # aNGS_flux is provided in photons/s
        aNGS_frameflux = aNGS_flux / aNGS_freq
        asigma = aNGS_FWHM_mas/sigmaToFWHM/self.mediumPixelScale
               
        xCoords = np.asarray(np.linspace(-self.largeGridSize/2.0+0.5, self.largeGridSize/2.0-0.5, self.largeGridSize), dtype=np.float32)
        yCoords = np.asarray(np.linspace(-self.largeGridSize/2.0+0.5, self.largeGridSize/2.0-0.5, self.largeGridSize), dtype=np.float32)
        xGrid, yGrid = np.meshgrid( xCoords, yCoords, sparse=False, copy=True)
                
        g2d = simple2Dgaussian( xGrid, yGrid, 0, 0, asigma)
        g2d = g2d * 1 / np.sum(g2d)
        I_k_data = g2d * aNGS_EE # Encirceld Energy in double FWHM is used to scale the PSF model
        I_k_data = I_k_data * aNGS_flux/aNGS_freq
            
        g2d_prime = simple2Dgaussian( xGrid, yGrid, self.p_offset, 0, asigma)
        g2d_prime = g2d_prime * 1 / np.sum(g2d_prime)       
        I_k_prime_data = g2d_prime * aNGS_EE # Encirceld Energy in double FWHM is used to scale the PSF model
        I_k_prime_data = I_k_prime_data * aNGS_flux/aNGS_freq
        
        I_k_data = intRebin(I_k_data, self.mediumShape) * self.downsample_factor**2
        I_k_prime_data = intRebin(I_k_prime_data,self.mediumShape) * self.downsample_factor**2
        W_Mask = np.zeros(self.mediumShape)
        # this is the array with the CoG weights, they are not normalized and so the variance is in pixel2
        ffx = np.arange(-self.mediumGridSize/2, self.mediumGridSize/2, 1.0) + 0.5
        (fx, fy) = np.meshgrid(ffx, ffx)
        # binary mask
        W_Mask = np.where( np.logical_or(fx**2 +fy**2 > WindowRadiusWCoG_LO**2, fx**2 + fy**2 < 0**2), 0.0, 1.0)
        ii1, ii2 = int(self.mediumGridSize/2-self.smallGridSize), int(self.mediumGridSize/2+self.smallGridSize)
        I_k_data = I_k_data[ii1:ii2,ii1:ii2]
        I_k_prime_data = I_k_prime_data[ii1:ii2,ii1:ii2]
        W_Mask = W_Mask[ii1:ii2,ii1:ii2]
        fx = fx[ii1:ii2,ii1:ii2]
        fy = fy[ii1:ii2,ii1:ii2]
        mu_ktr_array, var_ktr_array, sigma_ktr_array = self.meanVarSigma(I_k_data)
        mu_ktr_prime_array, var_ktr_prime_array, sigma_ktr_prime_array = self.meanVarSigma(I_k_prime_data)
        masked_mu0 = W_Mask * mu_ktr_array
        masked_mu = W_Mask * mu_ktr_prime_array
        masked_sigma = W_Mask**2 * var_ktr_array
        # TODO is the normalization correct?
        mux = np.sum(masked_mu*fx)/np.sum(masked_mu)
        muy = np.sum(masked_mu*fy)/np.sum(masked_mu)
        varx = np.sum(masked_sigma*fx**2)/(np.sum(masked_mu0)**2)
        vary = np.sum(masked_sigma*fy**2)/(np.sum(masked_mu0)**2)
        
        bias = mux/(self.p_offset/self.downsample_factor)

        return (bias,(mux,muy),(varx,vary))

    
    def computeTurbPSDs(self, fmin, fmax, freq_samples):    
        paramAndRange = ( 'f', fmin, fmax, freq_samples, 'linear' )
        scaleFactor = (500/2.0/np.pi)**2 # from rad**2 to nm**2

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
        psd_tip_turb = zplot1*scaleFactor
        xplot1, zplot1 = self.mIt.IntegralEvalE(self.sTurbPSDTilt, [paramAndRange], [(self.psdIntegrationPoints, 'linear')], 'rect')
        psd_tilt_turb = zplot1*scaleFactor
        return psd_tip_turb, psd_tilt_turb

    def computeFocusPSDs(self, fmin, fmax, freq_samples, alib):    
        paramAndRange = ( 'f', fmin, fmax, freq_samples, 'linear' )
        scaleFactor = (500/2.0/np.pi)**2 # from rad**2 to nm**2

        if self.computationPlatform=='GPU':
            xplot1, zplot1 = self.mIt.IntegralEvalE(self.sTurbPSDFocus, [paramAndRange], [(self.psdIntegrationPoints, 'linear')], 'rect')
        else:
            pool_size = int( min( mp.cpu_count(), freq_samples) )
            pool = mp.Pool(processes=pool_size)
            fx = np.linspace(fmin, fmax, freq_samples)
            paramsAndRangesG = [( 'f', fxx, 0.0, 0.0, 'provided' ) for fxx in fx]
            pool_outputs = pool.map(functools.partial(self.mIt.IntegralEvalE, eq=self.sTurbPSDFocus, integrationVarsSamplingSchemes=[(self.psdIntegrationPoints, 'linear')], method='rect') , [paramAndRange])
            pool.close()
            pool.join()
            for rr in pool_outputs:
                xplot1.append(rr[0])
                zplot1.append(rr[1])

        #print('x,z:', len(xplot1), len(zplot1))
        psd_freq = xplot1[0]
        psd_focus_turb = zplot1*scaleFactor
        
        psd_focus_sodium_lambda1 = lambdifyByName( self.sSodiumPSDFocus.rhs, ['f'], alib)
        psd_focus_sodium = psd_focus_sodium_lambda1( cp.array(psd_freq)) 
        return psd_focus_turb, psd_focus_sodium

    def checkStability(self,keys,values,TFeq):
        # substitute values in sympy expression
        dictTf = {'d':self.loopDelaySteps_LO}
        for key, value in zip(keys,values):
            dictTf[key] = value
        zTFeq = subsParamsByName(TFeq, dictTf)
        # compute numerator and denominator of the polynomials
        n,d = sp.fraction(sp.simplify(zTFeq))
        # create a transfer function
        z = sp.symbols('z', real=False)
        zTF = TransferFunction(n,d,z)
        # check stability thanks to the values of the poles
        if np.max(np.abs(zTF.poles())) > 0.99:
            return 0
        else:
            return 1
        
    def computeNoiseResidual(self, fmin, fmax, freq_samples, varX, bias, alib):
        npoints = 99
        Cfloat = self.fCValue.evalf()
        psd_tip_turb, psd_tilt_turb = self.computeTurbPSDs(fmin, fmax, freq_samples)
        psd_freq = np.asarray(np.linspace(fmin, fmax, freq_samples))
        
        if self.plot4debug:
            fig, ax1 = plt.subplots(1,1)
            im = ax1.plot(psd_freq,psd_tip_turb) 
            im = ax1.plot(psd_freq,psd_tilt_turb) 
            ax1.set_xscale('log')
            ax1.set_yscale('log')
            ax1.set_title('Turbulence PSD', color='black')
            ax1.set_xlabel('frequency [Hz]')
            ax1.set_ylabel('Power')
        
        df = psd_freq[1]-psd_freq[0]
        Df = psd_freq[-1]-psd_freq[0]
        sigma2Noise =  cpuArray(varX) / bias**2 * Cfloat / (Df / df)
        # must wait till this moment to substitute the noise level
        self.fTipS1 = subsParamsByName(self.fTipS_LO, {'phi^noise_Tip': sigma2Noise})
        self.fTiltS1 = subsParamsByName( self.fTiltS_LO, {'phi^noise_Tilt': sigma2Noise})    
        self.fTipS_lambda1 = lambdifyByName( self.fTipS1, ['g^Tip_0', 'f', 'phi^wind_Tip'], alib)
        self.fTiltS_lambda1 = lambdifyByName( self.fTiltS1, ['g^Tilt_0', 'f', 'phi^wind_Tilt'], alib)
        if self.displayEquation:
            print('computeNoiseResidual')
            try:
                display(self.fTipS1)
            except:
                print('    no self.fTipS1')
            try:
                display(self.fTiltS1)
            except:
                print('    no self.fTiltS1')
        if alib==gpulib and gpuEnabled:
            xp = cp
            psd_freq = cp.asarray(psd_freq)
            psd_tip_turb = cp.asarray(psd_tip_turb)
            psd_tilt_turb = cp.asarray(psd_tilt_turb)        
        else:
            xp = np
        if self.LoopGain_LO == 'optimize':
            # add small values of gain to have a good optimization
            # when the noise level is high.
            g0 = (0.00000001,0.0000001,0.000001,0.00001,0.0001,0.001)
            g0g = xp.concatenate((xp.asarray( g0),xp.linspace(0.01, 0.8, npoints)))
        elif self.LoopGain_LO == 'test':
            g0g = xp.asarray( xp.linspace(0.01, 0.99, npoints) )
        else:
            # if gain is set no optimization is done and bias is not compensated
            g0 = (bias*self.LoopGain_LO,bias*self.LoopGain_LO)
            g0g = xp.asarray(g0)
        
        e1 = psd_freq.reshape((1,psd_freq.shape[0]))
        e2 = psd_tip_turb.reshape((1,psd_tip_turb.shape[0]))
        e3 = psd_tilt_turb.reshape((1,psd_tilt_turb.shape[0]))
        e4 = g0g.reshape((g0g.shape[0], 1))
        psd_freq_ext, psd_tip_turb_ext, psd_tilt_turb_ext, g0g_ext = xp.broadcast_arrays(e1, e2, e3, e4)
               
        if self.plot4debug:
            fig, ax2 = plt.subplots(1,1)
            for x in range(g0g.shape[0]):
                im = ax2.plot(cpuArray(psd_freq),(self.fTipS_lambda1( g0g_ext, psd_freq_ext, psd_tip_turb_ext).get())[x,:]) 
            ax2.set_xscale('log')
            ax2.set_yscale('log')
            ax2.set_title('residual turb. PSD', color='black')
            ax2.set_xlabel('frequency [Hz]')
            ax2.set_ylabel('Power')
        
        resultTip = xp.absolute((xp.sum(self.fTipS_lambda1( g0g_ext, psd_freq_ext, psd_tip_turb_ext), axis=(1)) ) )
        resultTilt = xp.absolute((xp.sum(self.fTiltS_lambda1( g0g_ext, psd_freq_ext, psd_tilt_turb_ext), axis=(1)) ) )
        minTipIdx = xp.where(resultTip == xp.nanmin(resultTip))
        minTiltIdx = xp.where(resultTilt == xp.nanmin(resultTilt))
        if self.verbose:
            print('    best tip & tilt gain (noise):', "%.3f" % g0g[minTipIdx[0][0]], "%.3f" % g0g[minTiltIdx[0][0]])
        if alib==gpulib and gpuEnabled:
            return cp.asnumpy(resultTip[minTipIdx[0][0]]), cp.asnumpy(resultTilt[minTiltIdx[0][0]])
        else:
            return (resultTip[minTipIdx[0][0]], resultTilt[minTiltIdx[0][0]])

    def computeFocusNoiseResidual(self, fmin, fmax, freq_samples, varX, bias, alib):
        npoints = 99
        Cfloat = self.fCfocusValue.evalf()
        psd_focus_turb, psd_focus_sodium = self.computeFocusPSDs(fmin, fmax, freq_samples, alib)
        psd_freq = np.asarray(np.linspace(fmin, fmax, freq_samples))

        if self.plot4debug:
            fig, ax1 = plt.subplots(1,1)
            im = ax1.plot(psd_freq,psd_focus_turb)
            im = ax1.plot(psd_freq,psd_focus_sodium)
            ax1.set_xscale('log')
            ax1.set_yscale('log')
            ax1.set_title('Turbulence and Sodium PSD', color='black')
            ax1.set_xlabel('frequency [Hz]')
            ax1.set_ylabel('Power')

        df = psd_freq[1]-psd_freq[0]
        Df = psd_freq[-1]-psd_freq[0]
        sigma2Noise =  cpuArray(varX) / bias**2 * Cfloat / (Df / df)
        # must wait till this moment to substitute the noise level
        self.fFocusS1 = subsParamsByName(self.fFocusS_LO, {'phi^noise_Tip': sigma2Noise})
        self.fFocusS_lambda1 = lambdifyByName( self.fFocusS1, ['g^Tip_0', 'f', 'phi^wind_Tip'], alib)
        if self.displayEquation:
            print('computeNoiseResidual')
            try:
                display(self.fFocusS1)
            except:
                print('    no self.fFocusS1')

        if alib==gpulib and gpuEnabled:
            xp = cp
            psd_freq = cp.asarray(psd_freq)
            psd_focus_turb = cp.asarray(psd_focus_turb)
        else:
            xp = np
        if self.LoopGain_Focus == 'optimize':
            # add small values of gain to have a good optimization
            # when the noise level is high.
            g0 = (0.00000001,0.0000001,0.000001,0.00001,0.0001,0.001)
            g0g = xp.concatenate((xp.asarray( g0),xp.linspace(0.01, 0.99, npoints)))
        elif self.LoopGain_Focus == 'test':
            g0g = xp.asarray( xp.linspace(0.01, 0.99, npoints) )
        else:
            # if gain is set no optimization is done and bias is not compensated
            g0 = (bias*self.LoopGain_Focus,bias*self.LoopGain_Focus)
            g0g = xp.asarray(g0)

        e1 = psd_freq.reshape((1,psd_freq.shape[0]))
        e2 = psd_focus_turb.reshape((1,psd_focus_turb.shape[0]))
        e3 = g0g.reshape((g0g.shape[0], 1))
        psd_freq_ext, psd_focus_turb_ext, g0g_ext = xp.broadcast_arrays(e1, e2, e3)

        if self.plot4debug:
            fig, ax2 = plt.subplots(1,1)
            for x in range(g0g.shape[0]):
                im = ax2.plot(cpuArray(psd_freq),(self.fFocusS_lambda1( g0g_ext, psd_freq_ext, psd_focus_turb_ext).get())[x,:])
            ax2.set_xscale('log')
            ax2.set_yscale('log')
            ax2.set_title('residual PSD', color='black')
            ax2.set_xlabel('frequency [Hz]')
            ax2.set_ylabel('Power')

        resultFocus = xp.absolute((xp.sum(self.fFocusS_lambda1( g0g_ext, psd_freq_ext, psd_focus_turb_ext), axis=(1)) ) )

        minFocusIdx = xp.where(resultFocus == xp.nanmin(resultFocus))
        if self.verbose:
            print('    best focus gain (noise):',"%.3f" % cpuArray(g0g[minFocusIdx[0][0]]))
        if alib==gpulib and gpuEnabled:
            return cp.asnumpy(resultFocus[minFocusIdx[0][0]])
        else:
            return (resultFocus[minFocusIdx[0][0]])

    def computeWindResidual(self, psd_freq, psd_tip_wind0, psd_tilt_wind0, var1x, bias, alib):
        npoints = 99
        Cfloat = self.fCValue.evalf()
        df = psd_freq[1]-psd_freq[0]
        Df = psd_freq[-1]-psd_freq[0]
        psd_tip_wind = psd_tip_wind0 * df
        psd_tilt_wind = psd_tilt_wind0 * df

        if self.plot4debug:
            fig, ax1 = plt.subplots(1,1)
            im = ax1.plot(psd_freq,psd_tip_wind) 
            im = ax1.plot(psd_freq,psd_tilt_wind) 
            ax1.set_xscale('log')
            ax1.set_yscale('log')
            ax1.set_title('wind shake PSD', color='black')
            ax1.set_xlabel('frequency [Hz]')
            ax1.set_ylabel('Power')

        sigma2Noise = cpuArray(var1x) / bias**2 * Cfloat / (Df / df)

        if self.plot4debug:
            dict1 = {'d':self.loopDelaySteps_LO, 'f_loop':self.SensorFrameRate_LO}
            RTFwind = subsParamsByName( self.fTipS1tfW, dict1)
            NTFwind = subsParamsByName( self.fTipS1tfN, dict1)
            RTFwind_lambda1 = lambdifyByName( RTFwind, ['g^Tip_0', 'g^Tip_1', 'f'], cpulib)
            NTFwind_lambda1 = lambdifyByName( NTFwind, ['g^Tip_0', 'g^Tip_1', 'f'], cpulib)
            RTFwindL1 = RTFwind_lambda1( 0.25, 1.0, psd_freq)
            NTFwindL1 = NTFwind_lambda1( 0.25, 1.0, psd_freq)

            fig, ax2 = plt.subplots(1,1)
            im = ax2.plot(cpuArray(psd_freq),np.abs(RTFwindL1))
            im = ax2.plot(cpuArray(psd_freq),np.abs(NTFwindL1))
            ax2.set_xscale('log')
            ax2.set_yscale('log')
            ax2.set_title('TF', color='black')
            ax2.set_xlabel('frequency [Hz]')
            ax2.set_ylabel('Amplitude')
        if self.LoopGain_LO == 'optimize' or self.LoopGain_LO == 'test':
            self.fTipS1 = subsParamsByName(self.fTipS, {'phi^noise_Tip': sigma2Noise})
            self.fTiltS1 = subsParamsByName( self.fTiltS, {'phi^noise_Tilt': sigma2Noise})
            self.fTipS_lambda1 = lambdifyByName( self.fTipS1, ['g^Tip_0', 'g^Tip_1', 'f', 'phi^wind_Tip'], alib)
            self.fTiltS_lambda1 = lambdifyByName( self.fTiltS1, ['g^Tilt_0', 'g^Tilt_1', 'f', 'phi^wind_Tilt'], alib)
        else:
            self.fTipS1 = subsParamsByName(self.fTipS_LO, {'phi^noise_Tip': sigma2Noise})
            self.fTiltS1 = subsParamsByName( self.fTiltS_LO, {'phi^noise_Tilt': sigma2Noise})
            self.fTipS_lambda1 = lambdifyByName( self.fTipS1, ['g^Tip_0', 'f', 'phi^wind_Tip'], alib)
            self.fTiltS_lambda1 = lambdifyByName( self.fTiltS1, ['g^Tilt_0', 'f', 'phi^wind_Tilt'], alib)
        if alib==gpulib and gpuEnabled:
            xp = cp
            psd_freq = cp.asarray(psd_freq)
            psd_tip_wind = cp.asarray(psd_tip_wind)
            psd_tilt_wind = cp.asarray(psd_tilt_wind)        
        else:
            xp = np
        
        if self.LoopGain_LO == 'optimize':
            # add small values of gain to have a good optimization
            # when the noise level is high.
            g0 = (0.00000001,0.0000001,0.000001,0.00001,0.0001,0.001)
            g0g = xp.concatenate((xp.asarray( g0),xp.linspace(0.01, 0.8, npoints)))
            g0g, g1g = xp.meshgrid( g0g,g0g )
        elif self.LoopGain_LO == 'test':
            g0g = xp.asarray( xp.linspace(0.01, 0.99, npoints) )
            g0g, g1g = xp.meshgrid( g0g,g0g )
        else:
            # if gain is set no optimization is done and bias is not compensated
            g0 = (bias*self.LoopGain_LO,bias*self.LoopGain_LO)
            g0g = xp.asarray(g0)
        
        if self.LoopGain_LO == 'optimize' or self.LoopGain_LO == 'test':
            e1 = psd_freq.reshape((1,1,psd_freq.shape[0]))
            e2 = psd_tip_wind.reshape((1,1,psd_tip_wind.shape[0]))
            e3 = psd_tilt_wind.reshape((1,1,psd_tilt_wind.shape[0]))
            e4 = g0g.reshape((g0g.shape[0],g0g.shape[1],1))
            e5 = g1g.reshape((g1g.shape[0],g1g.shape[1],1))
            psd_freq_ext, psd_tip_wind_ext, psd_tilt_wind_ext, g0g_ext, g1g_ext  = xp.broadcast_arrays(e1, e2, e3, e4, e5)
            resultTip = xp.absolute((xp.sum(self.fTipS_lambda1( g0g_ext, g1g_ext, psd_freq_ext, psd_tip_wind_ext), axis=(2)) ) )
            resultTilt = xp.absolute((xp.sum(self.fTiltS_lambda1( g0g_ext, g1g_ext, psd_freq_ext, psd_tilt_wind_ext), axis=(2)) ) )
            if self.plot4debug:
                fig, ax2 = plt.subplots(1,1)
                for x in range(g0g.shape[0]):
                    im = ax2.plot(cpuArray(psd_freq),\
                        (self.fTipS_lambda1( g0g_ext, g1g_ext, psd_freq_ext, psd_tip_wind_ext).get())[x,int(npoints/2),:]) 
                ax2.set_xscale('log')
                ax2.set_yscale('log')
                ax2.set_title('residual wind PSD', color='black')
                ax2.set_xlabel('frequency [Hz]')
                ax2.set_ylabel('Power')
        else:
            e1 = psd_freq.reshape((1,psd_freq.shape[0]))
            e2 = psd_tip_wind.reshape((1,psd_tip_wind.shape[0]))
            e3 = psd_tilt_wind.reshape((1,psd_tilt_wind.shape[0]))
            e4 = g0g.reshape((g0g.shape[0], 1))
            psd_freq_ext, psd_tip_wind_ext, psd_tilt_wind_ext, g0g_ext  = xp.broadcast_arrays(e1, e2, e3, e4)
            resultTip = xp.absolute((xp.sum(self.fTipS_lambda1( g0g_ext, psd_freq_ext, psd_tip_wind_ext), axis=(1)) ) )
            resultTilt = xp.absolute((xp.sum(self.fTiltS_lambda1( g0g_ext, psd_freq_ext, psd_tilt_wind_ext), axis=(1)) ) )
            if self.plot4debug:
                fig, ax2 = plt.subplots(1,1)
                for x in range(g0g.shape[0]):
                    im = ax2.plot(cpuArray(psd_freq),(self.fTipS_lambda1( g0g_ext, psd_freq_ext, psd_tip_wind_ext).get())[x,:]) 
                ax2.set_xscale('log')
                ax2.set_yscale('log')
                ax2.set_title('residual wind PSD', color='black')
                ax2.set_xlabel('frequency [Hz]')
                ax2.set_ylabel('Power')
                
        minTipIdx = xp.where(resultTip == xp.nanmin(resultTip))
        minTiltIdx = xp.where(resultTilt == xp.nanmin(resultTilt))
        
        if self.verbose:
            if self.LoopGain_LO == 'optimize' or self.LoopGain_LO == 'test':
                print('    best tip & tilt gain (wind)',"%.3f" % cpuArray(g0g[minTipIdx[0][0],minTipIdx[1][0]]*g1g[minTipIdx[0][0],minTipIdx[1][0]]),\
                                                        "%.3f" % cpuArray(g0g[minTiltIdx[0][0],minTiltIdx[1][0]]*g1g[minTipIdx[0][0],minTipIdx[1][0]]))
            else:
                print('    best tip & tilt gain (wind)',"%.3f" % cpuArray(g0g[minTipIdx[0][0]]), "%.3f" % cpuArray(g0g[minTiltIdx[0][0]]))
                    
        if self.LoopGain_LO == 'optimize' or self.LoopGain_LO == 'test':
            if alib==gpulib and gpuEnabled:
                return cp.asnumpy(resultTip[minTipIdx[0][0], minTipIdx[1][0]]), cp.asnumpy(resultTilt[minTiltIdx[0][0], minTiltIdx[1][0]])
            else:
                return (resultTip[minTipIdx[0][0], minTipIdx[1][0]], resultTilt[minTiltIdx[0][0], minTiltIdx[1][0]])
        else:
            if alib==gpulib and gpuEnabled:
                return cp.asnumpy(resultTip[minTipIdx[0][0]]), cp.asnumpy(resultTilt[minTiltIdx[0][0]])
            else:
                return (resultTip[minTipIdx[0][0]], resultTilt[minTiltIdx[0][0]])

        
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
        
        _idx0 = {2:np.arange(0, 2*nstars, 2), 3:np.arange(1, 2*nstars, 2)}

        for ii in [2,3]:
            for jj in [2,3]:
                outputArray1 = self.covValue(ii, jj, inputsArray, hh)
                for pind in range(points):
                    for hidx, h_weight in enumerate(self.Cn2Weights):
                        matCasValue[ii-2+pind*2][_idx0[jj]] +=  h_weight*outputArray1[pind:nstars*points:points, hidx]
                        if pind==0:
                            matCssValue[ xp.ix_(_idx0[ii], _idx0[jj]) ] +=  xp.reshape( h_weight*outputArray1[nstars*points:, hidx], (nstars,nstars))
        return scaleF*matCaaValue, scaleF*matCasValue, scaleF*matCssValue

    def computeFocusCovMatrices(self, aCartPointingCoords, aCartNGSCoords, xp=np):
        if len(aCartPointingCoords.shape) > 1:
            points = aCartPointingCoords.shape[0]
        else:
            points = 1
        nstars = aCartNGSCoords.shape[0]        
        scaleF = (500.0/(2*np.pi))**2
        matCasValue = xp.zeros((points,nstars), dtype=xp.float32)
        matCssValue = xp.zeros((nstars,nstars), dtype=xp.float32)
        matCaaValue = self.covValue(4, 4, xp.asarray([1e-10, 1e-10]), xp.asarray([1]))[0,0]
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
        
        _idx0 = {4:np.arange(0, nstars, 1)}

        for ii in [4]:
            for jj in [4]:
                outputArray1 = self.covValue(4, 4, inputsArray, hh)
                for pind in range(points):
                    for hidx, h_weight in enumerate(self.Cn2Weights):
                        matCasValue[ii-4+pind][_idx0[jj]] +=  h_weight*outputArray1[pind:nstars*points:points, hidx]
                        if pind==0:
                            matCssValue[ xp.ix_(_idx0[ii], _idx0[jj]) ] +=  xp.reshape( h_weight*outputArray1[nstars*points:, hidx], (nstars,nstars))

        return scaleF*matCaaValue, scaleF*matCasValue, scaleF*matCssValue


    def multiCMatAssemble(self, aCartPointingCoordsV, aCartNGSCoords, aCnn, aC1):
        xp = np
        points = aCartPointingCoordsV.shape[0]
        Ctot = np.zeros((2*points,2))
        R, RT = self.buildReconstuctor2(aCartPointingCoordsV, aCartNGSCoords)
        Caa, Cas, Css = self.computeCovMatrices(xp.asarray(aCartPointingCoordsV), xp.asarray(aCartNGSCoords), xp=np)
        for i in range(points):
            Ri = R[2*i:2*(i+1),:]
            RTi = RT[:, 2*i:2*(i+1)]
            Casi = Cas[2*i:2*(i+1),:]
            C2b = xp.dot(Ri, xp.dot(Css, RTi)) - xp.dot(Casi, RTi) - xp.dot(Ri, Casi.transpose())
            C3 = xp.dot(Ri, xp.dot(xp.asarray(aCnn), RTi))
            # tomography (C2), noise (C3), wind (aC1) errors
            if self.noNoise:
                ss = xp.asarray(aC1) + Caa + C2b 
                print('    WARNING: LO noise is not active!')
            else:
                ss = xp.asarray(aC1) + Caa + C2b + C3
            Ctot[2*i:2*(i+1),:] = ss
            if self.verbose:
                print('    Star coordinates [arcsec]: ', ("{:.1f}, "*len(aCartPointingCoordsV[i])).format(*aCartPointingCoordsV[i]))
                print('    Total Cov. (tomo., tur.+noi., wind) [nm]:',"%.2f" % np.sqrt(np.trace(ss)),
                      '(', "%.2f" % np.sqrt(np.trace(Caa + C2b)), ',', "%.2f" % np.sqrt(np.trace(C3)), ',', "%.2f" % np.sqrt(np.trace(aC1)),')')
        return Ctot

        
    def CMatAssemble(self, aCartPointingCoordsV, aCartNGSCoords, aCnn, aC1):
        R, RT = self.buildReconstuctor2(np.asarray(aCartPointingCoordsV), aCartNGSCoords)
        Caa, Cas, Css = self.computeCovMatrices(np.asarray(aCartPointingCoordsV), aCartNGSCoords)        
        C2 = Caa + np.dot(R, np.dot(Css, RT)) - np.dot(Cas, RT) - np.dot(R, Cas.transpose())
        C3 = np.dot(R, np.dot(aCnn, RT))
        # sum tomography (C2), noise (C3), wind (aC1) errors
        if self.noNoise:
            ss = aC1 + C2
            print('    WARNING: LO noise is not active!')
        else:
            Ctot = aC1 + C2 + C3
        if self.verbose:
            print('    Star coordinates [arcsec]: ', ("{:.1f}, "*len(aCartPointingCoordsV)).format(*aCartPointingCoordsV))
            print('    Total Cov. (tomo., tur.+noi., wind) [nm]:', "%.2f" % np.sqrt(np.trace(Ctot)),
                  '(', "%.2f" % np.sqrt(np.trace(C2)), ',', "%.2f" % np.sqrt(np.trace(C3)), ',', "%.2f" % np.sqrt(np.trace(aC1)),')')
        return Ctot

    def multiFocusCMatAssemble(self, aCartNGSCoords, Cnn):
        xp = np
        Caa, Cas, Css = self.computeFocusCovMatrices(np.asarray((0,0)), np.asarray(aCartNGSCoords), xp=np)
        # NGS Rec. Mat. - MMSE estimator
        IMt = np.array(np.repeat(1, aCartNGSCoords.shape[0]))
        cov_turb_inv = np.array(1e-3) # the minimum ratio between turb. and noise cov. is 1e3 (this guarantees that the sum of the elements of R is 1).
        cov_noise = np.diag(np.clip(np.diag(Cnn),np.max(Css)*1e-2,np.max(Cnn))/np.max(Cnn)) # it clips noise covariance when noise level is low
        cov_noise_inv = np.linalg.pinv(cov_noise)
        H = np.matmul(np.matmul(IMt,cov_noise_inv),np.transpose(IMt))
        R = np.matmul(1/H*IMt,cov_noise_inv)
        RT = R.transpose()
        # sum tomography (Caa,Cas,Css) and noise (Cnn) errors for a on-axis star
        #Ctot = Caa + np.dot(R, np.dot(Css, RT)) - np.dot(Cas, RT) - np.dot(R, Cas.transpose()) + np.dot(R, np.dot(Cnn, RT)
        C2 = Caa + np.dot(R, np.dot(Css, RT)) - np.dot(Cas, RT) - np.dot(R, Cas.transpose())
        C3 = np.dot(R, np.dot(Cnn, RT))

        return C2, C3 

    def computeTotalResidualMatrixI(self, indices, aCartPointingCoords, aCartNGSCoords, aNGS_flux):
        nPointings = aCartPointingCoords.shape[0]
        maxFluxIndex = np.where(aNGS_flux==np.amax(aNGS_flux))
        nNaturalGS = len(indices)
        C1 = np.zeros((2,2))
        Cnn = np.zeros((2*nNaturalGS,2*nNaturalGS))
        if self.verbose:
            print('mavisLO.computeTotalResidualMatrix')
        for starIndex in range(nNaturalGS):
            bias, amu, avar = self.bias[indices[starIndex]], self.amu[indices[starIndex]], self.avar[indices[starIndex]]            
            nr = self.nr[indices[starIndex]] 
            wr = self.wr[indices[starIndex]] 
            if self.verbose:
                print('    NGS flux [ph/SA/s]       :', aNGS_flux[starIndex])
                print('    NGS coordinates [arcsec] : ', ("{:.1f}, "*len(aCartNGSCoords[starIndex])).format(*aCartNGSCoords[starIndex]))
                print('    turb. + noise residual (per NGS) [nm\u00b2]:',np.array(nr))
            Cnn[2*starIndex,2*starIndex] = nr[0]
            Cnn[2*starIndex+1,2*starIndex+1] = nr[1]
            if starIndex == maxFluxIndex[0][0]:
                C1[0,0] = wr[0]
                C1[1,1] = wr[1]
                if self.verbose:
                    print('    wind-shake residual (brightest NGS)    [nm\u00b2]:',np.array(wr))

        # C1 and Cnn do not depend on aCartPointingCoords[i]
        Ctot = self.multiCMatAssemble(aCartPointingCoords, aCartNGSCoords, Cnn, C1)
        return Ctot.reshape((nPointings,2,2))


    def computeTotalResidualMatrix(self, aCartPointingCoords, aCartNGSCoords, aNGS_flux, aNGS_freq, aNGS_SR, aNGS_EE, aNGS_FWHM_mas, doAll=True):
        self.bias = []
        self.amu = []
        self.avar = []
        self.wr = []
        self.nr = []
        nPointings = aCartPointingCoords.shape[0]
        maxFluxIndex = np.where(aNGS_flux==np.amax(aNGS_flux))
        nNaturalGS = aCartNGSCoords.shape[0]
        C1 = np.zeros((2,2))
        Cnn = np.zeros((2*nNaturalGS,2*nNaturalGS))

        if self.verbose:
            print('mavisLO.computeTotalResidualMatrix')
            
        for starIndex in range(nNaturalGS):
            self.configLOFreq( aNGS_freq[starIndex] )
            # one scalar (bias), two tuples of 2 (amu, avar)
            if self.simpleVarianceComputation:
                bias, amu, avar = self.simplifiedComputeBiasAndVariance(aNGS_flux[starIndex], aNGS_freq[starIndex], aNGS_EE[starIndex], aNGS_FWHM_mas[starIndex])
                # conversion from rad2 (peak-to-valley) to mas2
                # rad to OPD --> wavelength / (2pi), OPD to arcsec --> 3600*180/pi / D_SA, arcsec to mas --> 1000 and factor 4 from RMS to peak-to-valley
                if nNaturalGS != len(self.NumberLenslets):
                    NumberLenslets = self.NumberLenslets[0]
                else:
                    NumberLenslets = self.NumberLenslets[starIndex]
                rad2mas = self.SensingWavelength_LO*2000*206264.8/(np.pi*self.TelescopeDiameter/NumberLenslets)
                var1x = avar[0] * rad2mas**2
            else:
                bias, amu, avar = self.computeBiasAndVariance(aNGS_flux[starIndex], aNGS_freq[starIndex], aNGS_EE[starIndex], aNGS_FWHM_mas[starIndex])
                # conversion from pixel2 to mas2
                var1x = avar[0] * self.PixelScale_LO**2

            self.bias.append(bias)
            self.amu.append(amu)
            self.avar.append(avar)


            # noise propagation considering the number of sub-apertures and conversion from mas2 to nm2 is applied in computeNoiseResidual
            nr = self.computeNoiseResidual(0.25, self.maxLOtFreq, int(4*self.maxLOtFreq), var1x, bias, self.platformlib )

            # This computation is skipped if no wind shake PSD is present.
            if np.sum(self.psd_tip_wind) > 0 or np.sum(self.psd_tilt_wind) > 0:
                wr = self.computeWindResidual(self.psd_freq, self.psd_tip_wind, self.psd_tilt_wind, var1x, bias, self.platformlib )
            else:
                wr = (0,0)

            self.nr.append(nr)
            self.wr.append(wr)

            if self.verbose:
                print('    NGS flux [ph/SA/s]       :', aNGS_flux[starIndex])
                print('    NGS coordinates [arcsec] : ', ("{:.1f}, "*len(aCartNGSCoords[starIndex])).format(*aCartNGSCoords[starIndex]))
                print('    turb. + noise residual (per NGS) [nm\u00b2]:',np.array(nr))
                print('    wind-shake residual (per NGS)    [nm\u00b2]:',np.array(wr))
            Cnn[2*starIndex,2*starIndex] = nr[0]
            Cnn[2*starIndex+1,2*starIndex+1] = nr[1]
            if starIndex == maxFluxIndex[0][0]:
                C1[0,0] = wr[0]
                C1[1,1] = wr[1]

        if doAll:
            # C1 and Cnn do not depend on aCartPointingCoords[i]
            Ctot = self.multiCMatAssemble(aCartPointingCoords, aCartNGSCoords, Cnn, C1)
            return Ctot.reshape((nPointings,2,2))
        else:
            return None

    def computeFocusTotalResidualMatrixI(self, indices, aCartNGSCoords, aNGS_flux):
        nNaturalGS = len(indices)
        Cnn = np.zeros((nNaturalGS,nNaturalGS))
        if self.verbose:
            print('mavisLO.computeFocusTotalResidualMatrixI')
        for starIndex in range(nNaturalGS):
            bias, amu, avar = self.biasF[indices[starIndex]], self.amuF[indices[starIndex]], self.avarF[indices[starIndex]]            
            nr = self.nrF[indices[starIndex]] 
            if self.verbose:
                print('    NGS (focus sensor) flux [ph/SA/s]       :', aNGS_flux[starIndex])
                print('    NGS (focus sensor) coordinates [arcsec] : ', ("{:.1f}, "*len(aCartNGSCoords[starIndex])).format(*aCartNGSCoords[starIndex]))
                print('    turb. + noise residual (per NGS) [nm\u00b2]:',np.array(nr))
            Cnn[starIndex,starIndex] = nr
            
        C2, C3 = self.multiFocusCMatAssemble( aCartNGSCoords, Cnn)
            
        # difference
        CtotDiff = C2 + C3  - self.CtotL

        return CtotDiff

    def computeFocusTotalResidualMatrix(self, aCartNGSCoords, aNGS_flux, aNGS_freq, aNGS_SR, aNGS_EE, aNGS_FWHM_mas, doAll=True):
        self.biasF = []
        self.amuF = []
        self.avarF = []
        self.nrF = []
        
        maxFluxIndex = np.where(aNGS_flux==np.amax(aNGS_flux))
        nNaturalGS = aCartNGSCoords.shape[0]
        Cnn = np.zeros((nNaturalGS,nNaturalGS))
        
        if self.verbose:
            print('mavisLO.computeFocusTotalResidualMatrix')
            
        for starIndex in range(nNaturalGS):
            self.configFocusFreq( aNGS_freq[starIndex] )
            # one scalar (bias), two tuples of 2 (amu, avar)
            if self.simpleVarianceComputation or self.check_section_key('sensor_Focus'):
                bias, amu, avar = self.simplifiedComputeBiasAndVariance(aNGS_flux[starIndex], aNGS_freq[starIndex],
                                                                        aNGS_EE[starIndex], aNGS_FWHM_mas[starIndex], doLO=False)
                # conversion from rad2 (peak-to-valley) to mas2
                # rad to OPD --> wavelength / (2pi), OPD to arcsec --> 3600*180/pi / D_SA, arcsec to mas --> 1000 and factor 4 from RMS to peak-to-valley
                rad2mas = self.SensingWavelength_Focus*2000*206264.8/(np.pi*self.TelescopeDiameter/self.NumberLenslets_Focus[0])
                var1x = avar[0] * rad2mas**2
            else:
                bias, amu, avar = self.computeBiasAndVariance(aNGS_flux[starIndex], aNGS_freq[starIndex], aNGS_EE[starIndex], aNGS_FWHM_mas[starIndex])
                # conversion from pixel2 to mas2
                var1x = avar[0] * self.PixelScale_LO**2
                
            self.biasF.append(bias)
            self.amuF.append(amu)
            self.avarF.append(avar)
            
            # noise propagation for focus is 0.4 time the tilt one (linear)
            var1x *= 0.16
            # noise propagation considering the number of sub-apertures and conversion from mas2 to nm2 is applied in computeFocusNoiseResidual
            # except for the 0.16 factor that is applied above
            nr = self.computeFocusNoiseResidual(0.25, self.maxFocustFreq, int(4*self.maxFocustFreq), var1x, bias, self.platformlib )

            self.nrF.append(nr)

            Cnn[starIndex,starIndex] = nr
            if self.verbose:
                print('    NGS (focus sensor) flux [ph/SA/s]       :', aNGS_flux[starIndex])
                print('    NGS (focus sensor) coordinates [arcsec] : ', ("{:.1f}, "*len(aCartNGSCoords[starIndex])).format(*aCartNGSCoords[starIndex]))

        # reference error for LGS case
        HO_zen_field    = self.get_config_value('sources_HO','Zenith')
        HO_az_field     = self.get_config_value('sources_HO','Azimuth')
        HO_pointings = polarToCartesian(np.array( [HO_zen_field, HO_az_field]))
        aCartLGSCoords = np.dstack( (HO_pointings[0,:], HO_pointings[1,:]) ).reshape(-1, 2)
        # LGS Rec. Mat.
        RL = np.array(np.repeat(1, aCartLGSCoords.shape[0]))*1/np.float32(aCartLGSCoords.shape[0])
        RLT = RL.transpose()
        CaaL, CasL, CssL = self.computeFocusCovMatrices(np.asarray((0,0)), np.asarray(aCartLGSCoords), xp=np)
        # tomography error for a on-axis star for LGS WFSs
        self.CtotL = CaaL + np.dot(RL, np.dot(CssL, RLT)) - np.dot(CasL, RLT) - np.dot(RL, CasL.transpose())
        
        if doAll:
            C2, C3 = self.multiFocusCMatAssemble(aCartNGSCoords, Cnn)
            
            # difference
            CtotDiff = C2 + C3 - self.CtotL

            if self.verbose:
                print('    focus residual (tomo., tur.+noi., LGS) [nm]:', "%.2f" % np.sqrt(CtotDiff),
                    '(', "%.2f" % np.sqrt(C2), ',', "%.2f" % np.sqrt(C3), ',', "%.2f" % np.sqrt(self.CtotL),')')

            return CtotDiff
        else:
            return None

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
