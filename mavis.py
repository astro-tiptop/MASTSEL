from copy import deepcopy
import numpy as np
from astropy.io import fits
import matplotlib.pyplot as plt
from matplotlib import cm
from astropy.modeling import models, fitting


import scipy.signal
import scipy.ndimage
import scipy.interpolate
        
def polar_to_cart(polar_data, theta_step, range_step, x, y, order=3):
    from scipy.ndimage.interpolation import map_coordinates as mp
    X, Y = np.meshgrid(x, y)
    Tc = np.degrees(np.arctan2(Y, X)).ravel()
    Rc = (np.sqrt(X**2 + Y**2)).ravel()
    Tc[Tc < 0.0] = 360.0 + Tc[Tc < 0.0]
    Tc = Tc / theta_step
    Rc = Rc / range_step
    coords = np.vstack((Tc, Rc))
    polar_data = np.vstack((polar_data, polar_data[-1, :]))
    cart_data = mp(
        polar_data,
        coords,
        order=order,
        mode='constant',
        cval=np.nan)
    return(cart_data.reshape(len(y), len(x)).T)


def congrid(a, newdims, method='linear', centre=False, minusone=False):
    '''Arbitrary resampling of source array to new dimension sizes.
    Currently only supports maintaining the same number of dimensions.
    To use 1-D arrays, first promote them to shape (x,1).

    Uses the same parameters and creates the same co-ordinate lookup points
    as IDL''s congrid routine, which apparently originally came from a VAX/VMS
    routine of the same name.

    method:
    neighbour - closest value from original data
    nearest and linear - uses n x 1-D interpolations using
                         scipy.interpolate.interp1d
    (see Numerical Recipes for validity of use of n 1-D interpolations)
    spline - uses ndimage.map_coordinates

    centre:
    True - interpolation points are at the centres of the bins
    False - points are at the front edge of the bin

    minusone:
    For example- inarray.shape = (i,j) & new dimensions = (x,y)
    False - inarray is resampled by factors of (i/x) * (j/y)
    True - inarray is resampled by(i-1)/(x-1) * (j-1)/(y-1)
    This prevents extrapolation one element beyond bounds of input array.
    '''
    if a.dtype not in [np.float64, np.float32]:
        a = np.cast[float](a)

    m1 = np.cast[int](minusone)
    ofs = np.cast[int](centre) * 0.5
    old = np.array(a.shape)
    ndims = len(a.shape)
    if len(newdims) != ndims:
        print("[congrid] dimensions error. "
              "This routine currently only support "
              "rebinning to the same number of dimensions.")
        return None
    newdims = np.asarray(newdims, dtype=float)
    dimlist = []

    if method == 'neighbour':
        for i in range(ndims):
            base = np.indices(newdims)[i]
            dimlist.append((old[i] - m1) / (newdims[i] - m1)
                           * (base + ofs) - ofs)
        cd = np.array(dimlist).round().astype(int)
        newa = a[list(cd)]
        return newa

    elif method in ['nearest', 'linear']:
        # calculate new dims
        for i in range(ndims):
            base = np.arange(newdims[i])
            dimlist.append((old[i] - m1) / (newdims[i] - m1)
                           * (base + ofs) - ofs)
        # specify old dims
        olddims = [np.arange(i, dtype=np.float) for i in list(a.shape)]

        # first interpolation - for ndims = any
        mint = scipy.interpolate.interp1d(
            olddims[-1], a, kind=method, fill_value="extrapolate")
        newa = mint(dimlist[-1])

        trorder = [ndims - 1] + list(range(ndims - 1))
        for i in range(ndims - 2, -1, -1):
            newa = newa.transpose(trorder)

            mint = scipy.interpolate.interp1d(
                olddims[i], newa, kind=method, fill_value="extrapolate")
            newa = mint(dimlist[i])

        if ndims > 1:
            # need one more transpose to return to original dimensions
            newa = newa.transpose(trorder)

        return newa
    elif method in ['spline']:
        oslices = [slice(0, j) for j in old]
        oldcoords = np.ogrid[oslices]
        nslices = [slice(0, j) for j in list(newdims)]
        newcoords = np.mgrid[nslices]

        newcoords_dims = range(np.rank(newcoords))
        # make first index last
        newcoords_dims.append(newcoords_dims.pop(0))
        newcoords_tr = newcoords.transpose(newcoords_dims)
        # makes a view that affects newcoords

        newcoords_tr += ofs

        deltas = (np.asarray(old) - m1) / (newdims - m1)
        newcoords_tr *= deltas

        newcoords_tr -= ofs

        newa = scipy.ndimage.map_coordinates(a, newcoords)
        return newa
    else:
        print("Congrid error: Unrecognized interpolation type.\n",
              "Currently only \'neighbour\', \'nearest\',\'linear\',",
              "and \'spline\' are supported.")
        return None


from SYMAO.turbolence import *
from SYMAO.zernike import *

sp.init_printing(use_latex='mathjax')

degToRad = np.pi/180.0
radToDeg = 1.0/degToRad
radiansToArcsecs = 206265.0
arcsecsToRadians = 1.0/radiansToArcsecs
radiansToMas = radiansToArcsecs * 1000.0

sigmaToFWHM = 2 * np.sqrt(2.0 * np.log(2.0))
fit_window_max_size = 512
defaultArrayBackend = cp

#####TELESCOPE#####
TelescopeDiameter = 8.0 #[m]
DM_height = 10000 # [m]
MCAO_FoV = 120 #  (diameter)  [arcsec]
#####ATMOSPHERE#####
WindSpeed = 9.0 # [m/s]
r0_Value = 0.15 # [m]
L0_Value = 25.0 # [m]
Cn2_Value = [0.5, 0.5] # 
heights_Cn2 = [5000,15000, 10000] # [m]
#####SENSOR#####
#Sensor 2x2 subapertures N_sa,tot=4
SensingWavelength = 1650*1e-9 #[m]

SensorFrameRate = 500 # (= loop frequency): [500 Hz]  # , 250 Hz or 100 Hz (to be optimized)
#Corresponding delays (in frames):     [3]       # , 2, 1 
# dove sono usati? per ogni psf hai la FWHM in mas, con questi posso convertire in pixels
pixel_scale = 40 # [mas]
Npix_per_subap = 50 #
WindowRadiusWCoG = 2 # [pixels] calcolo sigma e mu, dimensione della finestra, cerchio di diametro 4
sigmaRON = 0.5 # [e-] NB: E'la sigma o la sigma**2: sigma ????????????????????????????????????????????????
ExcessNoiseFactor = 1.3 # nei calcoli precedentemente mettevamo 1.5, e'ok ?
Dark = 30 # [e-/s/pix]
skyBackground = 35 # [e-/s/pix]
ThresholdWCoG = 0 #
NewValueThrPix = 0

def cartesianToPolar(x):
    return np.asarray( [np.sqrt(x[0]**2+x[1]**2), np.arctan2(x[1],x[0])*radToDeg], dtype=np.float64 )
    
def polarToCartesian(x):
    return x[0] * np.asarray( [np.cos(x[1]*degToRad), np.sin(x[1]*degToRad)], dtype=np.float64 )

#####OTHERS#####
#PSD wind/vibration: the one for MAVIS (forest of peaks)
#Gains to be explored: linear vector from 0.01 to 0.99
#Directions for total jitter residual estimation (cartesian in arcsec): sono direzioni di puntamento, mappa 3x3
#[0,0], [5,-5], [5,5], [-5,5], [-5,-5], [15,-15], [15,15], [-15,15], [-15,-15]

cartTestPointingCoords = np.asarray([5,5])

#####NGS#####
#NGS coordinates (polar in [arcsec,degrees]): [30,0], [50,100],[10,240]
NGS_flux = [10000, 30000, 5000] # viene usato nel calcolo del rumore, fattore di normalizzazione della gaussian
NGS_SR_1650 = [0.4, 0.2, 0.6]
NGS_FWHM_mas = [90, 110, 85]

testNGSCoords = np.asarray([[30.0,0.0], [50.0,100.0],[10.0,240.0]])

cartTestNGSCoords0 = polarToCartesian(testNGSCoords[0]) #testNGSCoords[0][0] * np.asarray( [np.cos(testNGSCoords[0][1]*degToRad), np.sin(testNGSCoords[0][1]*degToRad)], dtype=np.float64 )  
cartTestNGSCoords1 = polarToCartesian(testNGSCoords[1]) #testNGSCoords[1][0] * np.asarray( [np.cos(testNGSCoords[1][1]*degToRad), np.sin(testNGSCoords[1][1]*degToRad)], dtype=np.float64 )  
cartTestNGSCoords2 = polarToCartesian(testNGSCoords[2]) #testNGSCoords[2][0] * np.asarray( [np.cos(testNGSCoords[2][1]*degToRad), np.sin(testNGSCoords[2][1]*degToRad)], dtype=np.float64 )  

cartTestNGSCoords = np.asarray([cartTestNGSCoords0, cartTestNGSCoords1, cartTestNGSCoords2])

def noisePropagationCoefficient():
    CC, D, Nsa = sp.symbols('C D N_sa\,tot', real=True, positive=True)
    expr0 = ((sp.pi/(180*3600*1000) * D / (4*1e-9)))**2/Nsa
    return sp.Eq(CC, expr0)

def noisePSDTip():
    f = sp.symbols('f', real=True, positive=True)
    phi_noise_tip = sp.Function( 'phi^noise_Tip')(f)
    sigma_w = sp.symbols('sigma^2_WCoG')
    mu_w = sp.symbols('mu_WCoG')
    df, DF, C = sp.symbols('df Delta_F C')
#    expr0 = noisePropagationCoefficient().rhs * sigma_w / (mu_w**2 * df * DF)
    expr0 = C * sigma_w / (mu_w**2 * df * DF)
    return sp.Eq(phi_noise_tip, expr0)

def noisePSDTilt():
    f = sp.symbols('f', real=True, positive=True)
    phi_noise_tilt = sp.Function( 'phi^noise_Tilt')(f)
    sigma_w = sp.symbols('sigma^2_WCoG')
    mu_w = sp.symbols('mu_WCoG')
    df, DF, C = sp.symbols('df Delta_F C')
#    expr0 = noisePropagationCoefficient().rhs * sigma_w / (mu_w**2 * df * DF)
    expr0 = C * sigma_w / (mu_w**2 * df * DF)
    return sp.Eq(phi_noise_tilt, expr0)

def turbPSDTip():
    f = sp.symbols('f', real=True, positive=True)
    phi_turb_tip = sp.Function( 'phi^turb_Tip')(f)    
    k, k_y, k_y_min, k_y_max, r0, L0, k0, V, R = sp.symbols('k k_y k_y_min k_y_max r_0 L_0 k_0 V R', positive=True)
    _lhs = sp.Function("P_phi")(k)
    k0 = 1.0 / L0 # 2 * sp.pi / L0
    with sp.evaluate(False):
        _rhs = 0.0229 * r0**(-sp.S(5) / sp.S(3)) * (k**2 + k0**2) ** (-sp.S(11) / sp.S(6))
    exprW = _rhs
    # expr1 = sp.sqrt((2*sp.pi*f/V)**2 + k_y**2)
    expr_k = sp.sqrt((f/V)**2 + k_y**2)
    # expr2 = sp.Integral( 8/(V*sp.pi**2*R**2) * (1/(k/2*sp.pi)**2)*_rhs * (2*sp.pi*f/(k*V))**2 * sp.besselj(2, R*k ), (k_y, k_y_min, k_y_max))
    expr2 = sp.Integral( 16/(V*sp.pi**2*R**2*k**2)*exprW * (f/(k*V))**2 * sp.besselj(2, 2*sp.pi*R*k )**2, (k_y, k_y_min, k_y_max))
    expr2 = expr2.subs({k:expr_k})
    return sp.Eq(phi_turb_tip, expr2)

def turbPSDTilt():
    f = sp.symbols('f', real=True, positive=True)
    phi_turb_tilt = sp.Function( 'phi^turb_Tilt')(f)    
    k, k_y, k_y_min, k_y_max, r0, L0, k0, V, R = sp.symbols('k k_y k_y_min k_y_max r_0 L_0 k_0 V R', positive=True)
    _lhs = sp.Function("P_phi")(k)
    k0 = 1.0 / L0 # 2 * sp.pi / L0
    with sp.evaluate(False):
        _rhs = 0.0229 * r0**(-sp.S(5) / sp.S(3)) * (k**2 + k0**2) ** (-sp.S(11) / sp.S(6))
    exprW = _rhs
    # expr1 = sp.sqrt((2*sp.pi*f/V)**2 + k_y**2)
    expr_k = sp.sqrt((f/V)**2 + k_y**2)
#    expr2 = sp.Integral( 8/(V*sp.pi**2*R**2) * (1/(k/2*sp.pi)**2)*_rhs * sp.sin(sp.acos( - 2*sp.pi*f/(k*V)) )**2 * sp.besselj(2, R*k ), (k_y, k_y_min, k_y_max))
    expr2 = sp.Integral( 16.0/(V*sp.pi**2*R**2*k**2) * exprW * (1.0 - (f/(k*V))**2) * sp.besselj(2, 2*sp.pi*R*k )**2, (k_y, k_y_min, k_y_max))
    expr2 = expr2.subs({k:expr_k})
    return sp.Eq(phi_turb_tilt, expr2)

def interactionMatrixNGS():
    N_NGS = sp.symbols('N_NGS', integer=True, positive=True)
    CC, H_DM, D, DP, r_FoV = sp.symbols('C H_DM D D\' r_FoV', real=True)
    expr0 = D + 2 * H_DM * r_FoV
    expr1 = 2 * H_DM * D / DP**2
    x_NGS, y_NGS = sp.symbols('x_NGS y_NGS', real=True)
    P = sp.Matrix([[1,0,CC*2*sp.S(sp.sqrt(3))*x_NGS, CC*sp.S(sp.sqrt(6))*y_NGS, CC*sp.S(sp.sqrt(6))*x_NGS], 
                    [0,1,CC*2*sp.S(sp.sqrt(3))*y_NGS, CC*sp.S(sp.sqrt(6))*x_NGS, -CC*sp.S(sp.sqrt(6))*y_NGS] ])
    P = P.subs({CC:expr1})
    P = P.subs({DP:expr0})
    return P

def residualTT():
    res, e_tip, e_tilt = sp.symbols('res epsilon_Tip epsilon_Tilt', real=True, positive=True)
    return sp.Eq(res, sp.sqrt(e_tip+e_tilt))

def residualTip():
    f, f_min, f_max = sp.symbols('f f_min f_max', real=True, positive=True)
    e_tip = sp.symbols('epsilon_Tip', real=True, positive=True)
#    phi_res_tip = sp.Function( 'phi^res_Tip')(f)
    phi_res_tip = sp.symbols( 'phi^res_Tip')
    return sp.Eq(e_tip, sp.Integral(phi_res_tip, (f, f_min, f_max)) )
        
def residualTilt():
    f, f_min, f_max = sp.symbols('f f_min f_max', real=True, positive=True)
    e_tilt = sp.symbols('epsilon_Tilt', real=True, positive=True)
#    phi_res_tilt = sp.Function( 'phi^res_Tilt')(f)
    phi_res_tilt = sp.symbols( 'phi^res_Tilt')
    return sp.Eq(e_tilt, sp.Integral(phi_res_tilt, (f, f_min, f_max)) )

def residualTipPSD():
#    phi_res_tip = sp.Function( 'phi^res_Tip')(f)
#    phi_wind_tip = sp.Function( 'phi^wind_Tip')(f)
#    phi_noise_tip = sp.Function( 'phi^noise_Tip')(f)
#    H_R_tip = sp.Function( 'H^R_Tip')(f)
#    H_N_tip = sp.Function( 'H^N_Tip')(f)
    phi_res_tip = sp.symbols( 'phi^res_Tip')
    phi_wind_tip = sp.symbols( 'phi^wind_Tip')
    phi_noise_tip = sp.symbols( 'phi^noise_Tip')
    H_R_tip = sp.symbols( 'H^R_Tip')
    H_N_tip = sp.symbols( 'H^N_Tip')
    phi_res_tip_expr = sp.Abs(H_R_tip)**2 * phi_wind_tip + sp.Abs(H_N_tip)**2 * phi_noise_tip
    return sp.Eq(phi_res_tip, phi_res_tip_expr)

def residualTiltPSD():
#    phi_res_tilt = sp.Function( 'phi^res_Tilt')(f)
#    phi_wind_tilt = sp.Function( 'phi^wind_Tilt')(f)
#    phi_noise_tilt = sp.Function( 'phi^noise_Tilt')(f)
#    H_R_tilt = sp.Function( 'H^R_Tilt')(f)
#    H_N_tilt = sp.Function( 'H^N_Tilt')(f)   
    phi_res_tilt = sp.symbols( 'phi^res_Tilt')
    phi_wind_tilt = sp.symbols( 'phi^wind_Tilt')
    phi_noise_tilt = sp.symbols( 'phi^noise_Tilt')
    H_R_tilt = sp.symbols( 'H^R_Tilt')
    H_N_tilt = sp.symbols( 'H^N_Tilt')
    phi_res_tilt_expr = sp.Abs(H_R_tilt)**2 * phi_wind_tilt + sp.Abs(H_N_tilt)**2 * phi_noise_tilt
    return sp.Eq(phi_res_tilt, phi_res_tilt_expr)

# 4 tf in z with 1 gain each to tune
def ztfTipWindMono():
    z = sp.symbols('z', real=False)
    H_R_tipz = sp.Function( 'H^R_Tip')(z)
    d = sp.symbols('d', integer=True)
    g_0_tip = sp.symbols('g^Tip_0', real=True)
    hrz_tip = (1-z**-1)/(1-z**-1+g_0_tip*z**-d)
    return sp.Eq(H_R_tipz, hrz_tip)

def ztfTiltWindMono():
    z = sp.symbols('z', real=False)
    H_R_tiltz = sp.Function( 'H^R_Tilt')(z)
    d = sp.symbols('d', integer=True)
    g_0_tilt = sp.symbols('g^Tilt_0', real=True)
    hrz_tilt = (1-z**-1)/(1-z**-1+g_0_tilt*z**-d)
    return sp.Eq(H_R_tiltz, hrz_tilt)

def ztfTipNoiseMono():
    z = sp.symbols('z', real=False)
    H_N_tipz = sp.Function( 'H^N_Tip')(z)    
    d = sp.symbols('d', integer=True)
    g_0_tip = sp.symbols('g^Tip_0', real=True)
    hnz_tip = g_0_tip*z**-d/(1-z**-1+g_0_tip*z**-d)
    return sp.Eq(H_N_tipz, hnz_tip)

def ztfTiltNoiseMono():
    z = sp.symbols('z', real=False)
    H_N_tiltz = sp.Function( 'H^N_Tilt')(z)    
    d = sp.symbols('d', integer=True)
    g_0_tilt = sp.symbols('g^Tilt_0', real=True)
    hnz_tilt = g_0_tilt*z**-d/(1-z**-1+g_0_tilt*z**-d)
    return sp.Eq(H_N_tiltz, hnz_tilt)

# end

# 4 tf in z with 2 gains each to tune

def ztfTipWind():
    z = sp.symbols('z', real=False)
    H_R_tipz = sp.Function( 'H^R_Tip')(z)
    d = sp.symbols('d', integer=True)
    g_0_tip, g_1_tip = sp.symbols('g^Tip_0 g^Tip_1', real=True)
    hrz_tip = (1-z**-1)**2/((1-z**-1+g_0_tip*z**-d)*(1-z**-1+g_1_tip))
    return sp.Eq(H_R_tipz, hrz_tip)

def ztfTiltWind():
    z = sp.symbols('z', real=False)
    H_R_tiltz = sp.Function( 'H^R_Tilt')(z)
    d = sp.symbols('d', integer=True)
    g_0_tilt, g_1_tilt = sp.symbols('g^Tilt_0 g^Tilt_1', real=True)
    hrz_tilt = (1-z**-1)**2/((1-z**-1+g_0_tilt*z**-d)*(1-z**-1+g_1_tilt))
    return sp.Eq(H_R_tiltz, hrz_tilt)

def ztfTipNoise():
    z = sp.symbols('z', real=False)
    H_N_tipz = sp.Function( 'H^N_Tip')(z)    
    d = sp.symbols('d', integer=True)
    g_0_tip, g_1_tip = sp.symbols('g^Tip_0 g^Tip_1', real=True)
    hnz_tip = g_0_tip*g_1_tip*z**-d/((1-z**-1+g_0_tip*z**-d)*(1-z**-1+g_1_tip))
    return sp.Eq(H_N_tipz, hnz_tip)

def ztfTiltNoise():
    z = sp.symbols('z', real=False)
    H_N_tiltz = sp.Function( 'H^N_Tilt')(z)
    d = sp.symbols('d', integer=True)
    g_0_tilt, g_1_tilt = sp.symbols('g^Tilt_0 g^Tilt_1', real=True)
    hnz_tilt = g_0_tilt*g_1_tilt*z**-d/((1-z**-1+g_0_tilt*z**-d)*(1-z**-1+g_1_tilt)) 
    return sp.Eq(H_N_tiltz, hnz_tilt )

# end

# 4 tf in f obtained from corresponding tf in z

def tfTipWind(ztf = ztfTipWind()):
    f, f_loop = sp.symbols('f f_loop', real=True, positive=True)
    hrz_tip = ztf.rhs
    hrf_tip = subsParamsByName(hrz_tip, {'z':sp.exp(2*sp.pi*f*sp.I/f_loop)})
    H_R_tip = sp.Function( 'H^R_Tip')(f)
    return sp.Eq(H_R_tip, hrf_tip)

def tfTiltWind(ztf = ztfTiltWind()):
    f, f_loop = sp.symbols('f f_loop', real=True, positive=True)
    hrz_tilt = ztf.rhs
    hrf_tilt = subsParamsByName(hrz_tilt, {'z':sp.exp(2*sp.pi*f*sp.I/f_loop)})
    H_R_tilt = sp.Function( 'H^R_Tilt')(f)
    return sp.Eq(H_R_tilt, hrf_tilt)

def tfTipNoise(ztf = ztfTipNoise()):
    f, f_loop = sp.symbols('f f_loop', real=True, positive=True)
    hnz_tip = ztf.rhs
    hnf_tip = subsParamsByName(hnz_tip, {'z':sp.exp(2*sp.pi*f*sp.I/f_loop)})
    H_N_tip = sp.Function( 'H^N_Tip')(f)
    return sp.Eq(H_N_tip, hnf_tip)

def tfTiltNoise(ztf = ztfTiltNoise()):
    f, f_loop = sp.symbols('f f_loop', real=True, positive=True)
    hnz_tilt = ztf.rhs
    hnf_tilt = subsParamsByName(hnz_tilt, {'z':sp.exp(2*sp.pi*f*sp.I/f_loop)})
    H_N_tilt = sp.Function( 'H^N_Tilt')(f)
    return sp.Eq(H_N_tilt, hnf_tilt)

# end

def completeIntegralTipLO():
    completeIntegralTipV = subsParamsByName( residualTip().rhs, {'phi^res_Tip':residualTipPSD().rhs} )
    completeIntegralTipV = subsParamsByName( completeIntegralTipV, {'H^R_Tip':tfTipWind(ztfTipWindMono()).rhs, 'H^N_Tip':tfTipNoise(ztfTipNoiseMono()).rhs} )
    return completeIntegralTipV

def completeIntegralTiltLO():
    completeIntegralTiltV = subsParamsByName( residualTilt().rhs, {'phi^res_Tilt':residualTiltPSD().rhs} )
    completeIntegralTiltV = subsParamsByName( completeIntegralTiltV, {'H^R_Tilt':tfTiltWind(ztfTiltWindMono()).rhs, 'H^N_Tilt':tfTiltNoise(ztfTiltNoiseMono()).rhs} )
    return completeIntegralTiltV

                        
def completeIntegralTip():
    completeIntegralTipV = subsParamsByName( residualTip().rhs, {'phi^res_Tip':residualTipPSD().rhs} )
    completeIntegralTipV = subsParamsByName( completeIntegralTipV, {'H^R_Tip':tfTipWind().rhs, 'H^N_Tip':tfTipNoise().rhs} )
    return completeIntegralTipV

def completeIntegralTilt():
    completeIntegralTiltV = subsParamsByName( residualTilt().rhs, {'phi^res_Tilt':residualTiltPSD().rhs} )
    completeIntegralTiltV = subsParamsByName( completeIntegralTiltV, {'H^R_Tilt':tfTiltWind().rhs, 'H^N_Tilt':tfTiltNoise().rhs} )
    return completeIntegralTiltV

# Covariance between zernike modes

def cov_expr_jk(expr, jj_value, kk_value):
    nj_value, mj_value = noll_to_zern(jj_value)
    nk_value, mk_value = noll_to_zern(kk_value)
    rexpr = subsParamsByName(expr, {'j': jj_value, 'k': kk_value, 'n_j': nj_value, 'm_j': abs(mj_value), 'n_k': nk_value, 'm_k': abs(mk_value)})
    return rexpr

def zernikeCovarianceD():
    f = sp.symbols('f', positive=True)
    hh, z1, z2 = sp.symbols('h z_1 z_2', positive=True)
    jj, nj, mj, kk, nk, mk = sp.symbols('j n_j m_j k n_k m_k', integer=True)
    R1, R2 = sp.symbols('R_1 R_2', positive=True)
    f0 = (-1)**mk * sp.sqrt((nj+1)*(nk+1)) * sp.I**(nj+nk) * 2 ** ( 1 - 0.5*((sp.KroneckerDelta(0,mj) + sp.KroneckerDelta(0,mk))) )
    f1 = 1 / (sp.pi * R1 * R2) 
    r0, L0, k0 = sp.symbols('r_0 L_0 k_0', positive=True)
    ff0 = 1 / L0
    with sp.evaluate(False):
        psd_def = 0.0229*r0**(-sp.S(5)/sp.S(3))*(f**2+ff0**2)**(-sp.S(11)/sp.S(6))
    f3 = sp.cos( (mj+mk)*theta + (sp.pi/4) * ( (1-sp.KroneckerDelta(0, mj)) * ((-1)**jj-1) + (1-sp.KroneckerDelta(0, mk)) * ((-1)**kk-1)) )
    f4 = I**(3*(mj+mk)) * sp.besselj( (mj+mk), 2*sp.pi*f*hh*rho)
    f5 = sp.cos( (mj-mk)*theta + (sp.pi/4) * ( (1-sp.KroneckerDelta(0, mj)) * ((-1)**jj-1) - (1-sp.KroneckerDelta(0, mk)) * ((-1)**kk-1)) )
    f6 = I**(3*sp.Abs(mj-mk)) * sp.besselj( sp.Abs(mj-mk), 2*sp.pi*f*hh*rho)    
    _rhs = f0 * f1 * (psd_def * sp.besselj( nj+sp.Integer(1), 2*sp.pi*f*R1) * sp.besselj( nj+sp.Integer(1), 2*sp.pi*f*R2) / f) * (f3*f4+f5*f6)    
    _lhs = sp.Function('dW_phi')(rho)
    return (_lhs, _rhs, sp.relational.Eq(_lhs, _rhs))

def expr_phi():
    x = sp.symbols('x', real=True)
    return (sp.S(1)/sp.sqrt(sp.S(2)*sp.pi)) * sp.exp( - x**2 / 2)

def expr_Phi():
    x = sp.symbols('x', real=True)
    return (sp.S(1)/sp.S(2)) * (1+sp.erf(x/sp.sqrt(sp.S(2))))

def expr_G():
    i = sp.symbols('i', integer=True, positive=True)
    F, z = sp.symbols('F z', real=True)
    return z ** (i/(F-1) -1 )  * sp.exp(-z/(F-1)) / ( sp.exp(sp.log(F-1)*(i/(F-1))) *  sp.gamma(i/(F-1)) )

def truncatedMeanBasic():
    i, i_max = sp.symbols('i i_max', integer=True)
    # f_k = I_k + back, vedi appedice D : back: 0.0
    f_k = sp.symbols('f_k', real=True)
    sigma_ron, b, t, nu = sp.symbols('sigma_RON b t nu', real=True)
    expr_K_i = sp.exp(-(f_k+b)) * (f_k+b) ** i / sp.factorial(i)
    f1 = subsParamsByName( expr_phi(), {'x': (t-(i-b))/sigma_ron})
    f2 = subsParamsByName( expr_Phi(), {'x': (i-b-t)/sigma_ron})
    f3 = subsParamsByName( expr_Phi(), {'x': (t-(i-b))/sigma_ron})
    fK = expr_K_i
    _rhs = sp.Sum(fK * ( sigma_ron * f1  + (i-b) * f2 + nu * f3 ) , (i, 0, i_max))
    _lhs = sp.symbols('mu_k\,thr')
    return (_lhs, _rhs, sp.relational.Eq(_lhs, _rhs))

def truncatedVarianceBasic():
    mu_k = sp.symbols('mu_k_thr')
    i, i_max = sp.symbols('i i_max', integer=True)
    f_k = sp.symbols('f_k', real=True)
    sigma_ron, b, t, nu = sp.symbols('sigma_RON b t nu', real=True)
    expr_K_i = sp.exp(-(f_k+b)) * (f_k+b) ** i / sp.factorial(i)
    f1 = subsParamsByName( expr_phi(), {'x': (t-(i-b))/sigma_ron})
    f2 = subsParamsByName( expr_Phi(), {'x': (i-b-t)/sigma_ron})
    f3 = subsParamsByName( expr_Phi(), {'x': (t-(i-b))/sigma_ron})
    fK = expr_K_i
    _rhs = sp.Sum(fK * ( sigma_ron * (t+i-b) * f1 + (sigma_ron**2 + (i-b)**2) * f2 + nu**2 * f3 ) , (i, 0, i_max)) - mu_k**2
    _lhs = sp.symbols('sigma^2_k\,thr')
    return (_lhs, _rhs, sp.relational.Eq(_lhs, _rhs))


def truncatedMeanIntegrand():
    F, z = sp.symbols('F z', real=True)
    i = sp.symbols('i', integer=True)
    f_k = sp.symbols('f_k', real=True)
    sigma_ron, b, t, nu = sp.symbols('sigma_RON b t nu', real=True)
    f4 = subsParamsByName(expr_phi(), {'x': (t-(z-b))/sigma_ron})
    f5 = subsParamsByName(expr_Phi(), {'x': (z-b-t)/sigma_ron})
    f6 = subsParamsByName(expr_Phi(), {'x': (t-(z-b))/sigma_ron})
    _rhs = expr_G() * ( sigma_ron * f4 + (z-b) * f5 + nu * f6 )
    _lhs = sp.symbols('I_mu_k\,thr')
    return (_lhs, _rhs, sp.relational.Eq(_lhs, _rhs))


def truncatedVarianceIntegrand():
    F, z = sp.symbols('F z', real=True)
    i = sp.symbols('i', integer=True)
    f_k = sp.symbols('f_k', real=True)
    sigma_ron, b, t, nu = sp.symbols('sigma_RON b t nu', real=True)
    f4 = subsParamsByName(expr_phi(), {'x': (t-(z-b))/sigma_ron})
    f5 = subsParamsByName(expr_Phi(), {'x': (z-b-t)/sigma_ron})
    f6 = subsParamsByName(expr_Phi(), {'x': (t-(z-b))/sigma_ron})
    _rhs = expr_G() * ( sigma_ron * (t+z-b) * f4 + (sigma_ron**2 + (z-b)**2) * f5 + nu**2 * f6 )
    _lhs = sp.symbols('I_sigma_k\,thr')
    return (_lhs, _rhs, sp.relational.Eq(_lhs, _rhs))


def truncatedMeanComponents():
    F, z = sp.symbols('F z', real=True)
    i = sp.symbols('i', integer=True, positive=True)
    f_k = sp.symbols('f_k', real=True)
    sigma_ron, b, t, nu = sp.symbols('sigma_RON b t nu', real=True)
    expr_K_i = sp.exp(-(f_k+b)) * (f_k+b) ** i / sp.factorial(i)
    f1 = subsParamsByName(expr_phi(), {'x': (t+b)/sigma_ron})
    f2 = subsParamsByName(expr_Phi(), {'x': -(t+b)/sigma_ron})
    f3 = subsParamsByName(expr_Phi(), {'x': (t+b)/sigma_ron})
    z_max = sp.symbols('z_max')
    expr10 = sp.exp(-(f_k+b)) * ( sigma_ron * f1  - b * f2 + nu * f3 )
    _integrand = subsParamsByName(truncatedMeanIntegrand()[1], {'z':z, 'i':i} )
    return (expr10, expr_K_i, sp.Integral( _integrand, (z, 0.0001, z_max)))


def truncatedMean():
    expr10, expr_K_i, integral =  truncatedMeanComponents()
    i_max = sp.symbols('i_max', integer=True, positive=True)
    i = getSymbolByName(expr_K_i, 'i')
    _rhs = expr10 + sp.Sum( expr_K_i *  integral , (i, 1, i_max) )
    _lhs = sp.symbols('mu_k\,thr')
    return (_lhs, _rhs, sp.relational.Eq(_lhs, _rhs))


def truncatedVarianceComponents():
    F, z = sp.symbols('F z', real=True)
    i = sp.symbols('i', integer=True, positive=True)
    f_k = sp.symbols('f_k', real=True)
    sigma_ron, b, t, nu = sp.symbols('sigma_RON b t nu', real=True)
    expr_K_i = sp.exp(-(f_k+b)) * (f_k+b) ** i / sp.factorial(i)
    f1 = subsParamsByName(expr_phi(), {'x': (t+b)/sigma_ron})
    f2 = subsParamsByName(expr_Phi(), {'x': -(t+b)/sigma_ron})
    f3 = subsParamsByName(expr_Phi(), {'x': (t+b)/sigma_ron})
    z_max = sp.symbols('z_max')
    expr20 = sp.exp(-(f_k+b)) * (sigma_ron * (t-b) * f1 + (sigma_ron**2 + b**2) * f2 + nu**2 * f3)
    _integrand = subsParamsByName(truncatedVarianceIntegrand()[1], {'z':z, 'i':i} )    
    return (expr20, expr_K_i, sp.Integral( _integrand, (z, 0.0001, z_max)))


def truncatedVariance():
    expr20, expr_K_i, integral =  truncatedVarianceComponents()
    i_max = sp.symbols('i_max', integer=True, positive=True)
    mu_k = sp.symbols('mu_k_thr')
    i = getSymbolByName(expr_K_i, 'i')
    _rhs = expr20 + sp.Sum(expr_K_i*integral , (i, 1, i_max)) - mu_k**2
    _lhs = sp.symbols('sigma^2_k\,thr')
    return (_lhs, _rhs, sp.relational.Eq(_lhs, _rhs))


MavisFormulas = Formulary("MAVIS",
                        ['ZernikeCovarianceD', 
                         'TruncatedMeanBasic', 
                         'TruncatedVarianceBasic',
                         'TruncatedMean', 
                         'TruncatedMeanIntegrand', 
                         'TruncatedVariance',
                         'TruncatedVarianceIntegrand',
                        ],
                        [zernikeCovarianceD(),
                        truncatedMeanBasic(), 
                        truncatedVarianceBasic(),
                        truncatedMean(), 
                        truncatedMeanIntegrand(), 
                        truncatedVariance(),
                        truncatedVarianceIntegrand(),
                        ] )


def FWHM_from_sigma(sigma):
    return sigma * sigmaToFWHM


def sigma_from_FWHM(fwhm):
    return fwhm / sigmaToFWHM


def hostData(_data):
    if defaultArrayBackend == cp:
        return cp.asnumpy(_data)
    else:
        return _data


def twoPsfsPlot(result, myResult):
    plt.xscale('linear')
    plt.yscale('log')
    plt.plot(result[:, result.shape[0] // 2])
    plt.plot(myResult[:, myResult.shape[0] // 2])
    plt.show()


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


def fitGaussian(image):
    N = image.shape[0]
    p_init = models.Gaussian2D(
        amplitude=np.max(image),
        x_mean=N / 2,
        y_mean=N / 2)
    fit_p = fitting.LevMarLSQFitter()
    return fit_p, p_init


def fitAiry(image):
    N = image.shape[0]
    p_init = models.AiryDisk2D(amplitude=np.max(image), x_0=N / 2, y_0=N / 2)
    fit_p = fitting.LevMarLSQFitter()
    return fit_p, p_init


def standardPsfPlot(img2):
    fig, axs = plt.subplots(1, 3, figsize=(12, 4), constrained_layout=True)
    s = img2.shape[0]
    axs[0].imshow(img2, cmap=cm.gist_heat)
    axs[1].plot(img2[s // 2, :])
    axs[2].plot(img2[:, s // 2])
    plt.show()

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
                            self.xp)))),
            self.xp)

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
