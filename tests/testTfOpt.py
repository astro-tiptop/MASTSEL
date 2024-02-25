from mastsel.mavisFormulas import *
from mastsel.mavisLO import *
from mastsel.mavisPsf import *

from scipy import optimize

path = "data/ini/"
parametersFile = 'mavisParamsTests'
mLO = MavisLO(path, parametersFile)
mLO.LoopGain_LO = 0.3


f1 = mLO.get_config_value('RTC','SensorFrameRate_LO')
ff= [f1, f1, f1]
mLO.configLOFreq(f1)
# WIND SHAKE PSD
#psd_freq, psd_tip_wind0, psd_tilt_wind0 = mLO.loadWindPsd('data/windpsd_mavis.fits')
psd_freq, psd_tip_wind0, psd_tilt_wind0 = mLO.loadWindPsd('data/morfeo_windshake8ms_psd_1k.fits')

# TIP and TILT PSDs
psd_tip_wind1, psd_tilt_wind1 = mLO.computeTurbPSDs(np.min(psd_freq)/10., np.max(psd_freq), psd_freq.shape[0])
psd_freq1 = np.asarray(np.linspace(np.min(psd_freq)/10., np.max(psd_freq), psd_freq.shape[0]))
# FOCUS PSDs        
psd_focus_wind, psd_focus_sodium = mLO.computeFocusPSDs(np.min(psd_freq)/10., np.max(psd_freq), psd_freq.shape[0], cpulib)

aNGS_flux = 50.*f1 # flux per second per sub-aperture
aNGS_freq = f1
aNGS_SR_LO = 0.5
aNGS_FWHM_mas = mLO.PixelScale_LO * 2.

bias, amu, avar = mLO.simplifiedComputeBiasAndVariance(aNGS_flux, aNGS_freq, aNGS_SR_LO, aNGS_FWHM_mas) 

var1x = avar[0] * mLO.PixelScale_LO**2

Cfloat = mLO.fCValue.evalf()
df = psd_freq[1]-psd_freq[0]
Df = psd_freq[-1]-psd_freq[0]
psd_tip_wind = psd_tip_wind0 * df
psd_tilt_wind = psd_tilt_wind0 * df
sigma2Noise = var1x / bias**2 * Cfloat / (Df / df)

mLO.fTipS1 = subsParamsByName(mLO.fTipS_LO, {'phi^noise_Tip': sigma2Noise})
mLO.fTipS_lambda1 = lambdifyByName( mLO.fTipS1, ['g^Tip_0', 'f', 'phi^wind_Tip'], cpulib)

xp = np
 
# optimization 
keys = ['g^Tip_0']
fun = lambda xx,yy: mLO.checkStability(keys,xx,mLO.fTipS_LO1ztfW) \
                    * xp.absolute((xp.sum(mLO.fTipS_lambda1(xx[0],yy[0],yy[1])) )) \
                    + (1-mLO.checkStability(keys,xx,mLO.fTipS_LO1ztfW))*10000
bounds = optimize.Bounds(lb=[1e-6], ub=[1], keep_feasible=False)
args = [psd_freq,psd_tip_wind]
x0 = [0.1]
res = optimize.minimize(fun,x0,args=args,bounds=bounds)
print(res)

# variables for display
dict1 = {'d':mLO.loopDelaySteps_LO, 'f_loop':f1}
RTFwind = subsParamsByName( mLO.fTipS_LO1tfW, dict1)
NTFwind = subsParamsByName( mLO.fTipS_LO1tfN, dict1)

RTFwind_lambda1 = lambdifyByName( RTFwind, ['g^Tip_0', 'f'], cpulib)
NTFwind_lambda1 = lambdifyByName( NTFwind, ['g^Tip_0', 'f'], cpulib)

RTFwindL1 = RTFwind_lambda1( res.x[0], psd_freq)
NTFwindL1 = NTFwind_lambda1( res.x[0], psd_freq)

outPsd = mLO.fTipS_lambda1(res.x[0],psd_freq,psd_tip_wind)

# plots

plt.figure(figsize=(12,9))
plt.xscale('log')
plt.yscale('log')
plt.plot(psd_freq,abs(RTFwindL1))
plt.plot(psd_freq,abs(NTFwindL1))
plt.xlabel('frequency [Hz]')
plt.ylabel('Amplitude')
plt.show(block=False)

plt.figure(figsize=(12,9))
plt.xscale('log')
plt.yscale('log')
plt.plot(psd_freq,psd_tip_wind0)
plt.plot(psd_freq1,psd_tip_wind1)
plt.plot(psd_freq1,psd_tilt_wind1)
plt.plot(psd_freq1,psd_focus_wind)
plt.xlabel('frequency [Hz]')
plt.ylabel('power [nm2/Hz]')
plt.show(block=False)

plt.figure(figsize=(12,9))
plt.xscale('log')
plt.yscale('log')
plt.plot(psd_freq1,psd_focus_wind)
plt.plot(psd_freq1,psd_focus_sodium)
plt.xlabel('frequency [Hz]')
plt.ylabel('power [nm2/Hz]')
plt.show(block=False)

plt.figure(figsize=(12,9))
plt.xscale('log')
plt.yscale('log')
plt.plot(psd_freq,psd_tip_wind0)
plt.plot(psd_freq,outPsd)
plt.xlabel('frequency [Hz]')
plt.ylabel('power [nm2/Hz]')
plt.show(block=False)
