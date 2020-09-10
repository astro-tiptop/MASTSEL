from SYMAO.turbolence import *
from SYMAO.zernike import *

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
