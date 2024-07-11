import sympy as sp
from symao.turbolence import *
from symao.zernike import *

def createMavisFormulary():

    # define a set of symbols and functions with unique representations
    
    rho = sp.symbols('rho', positive=True)
    theta = sp.symbols('theta', real=True)
    f, f_min, f_max, f_loop = sp.symbols('f f_min f_max f_loop', positive=True)
    z = sp.symbols('z', real=False)
    d = sp.symbols('d', integer=True)
    k, k_y, k_y_min, k_y_max = sp.symbols('k k_y k_y_min k_y_max', positive=True)
    r0, L0, k0, V, R = sp.symbols('r_0 L_0 k_0 V R', positive=True)
    C, D, Nsa = sp.symbols('C D N_sa_tot', real=True, positive=True)
    sigma_w = sp.symbols('sigma^2_WCoG')
    mu_w = sp.symbols('mu_WCoG')
    df, DF = sp.symbols('df Delta_F')
    zA = sp.symbols('ZenithAngle')

    # N_NGS = sp.symbols('N_NGS', integer=True, positive=True)
    H_DM, DP, r_FoV = sp.symbols('H_DM D\' r_FoV', real=True)
    x_NGS, y_NGS = sp.symbols('x_NGS y_NGS', real=True)

    phi_noise_tip_f = sp.Function( 'phi^noise_Tip^f')(f)
    phi_noise_tilt_f = sp.Function( 'phi^noise_Tilt^f')(f)
    phi_turb_tip_f = sp.Function( 'phi^turb_Tip^f')(f)    
    phi_turb_tilt_f = sp.Function( 'phi^turb_Tilt^f')(f)
    phi_turb_focus_f = sp.Function( 'phi^turb_Focus^f')(f)
    phi_sodium_focus_f = sp.Function( 'phi^sodium_Focus^f')(f)

    res, e_tip, e_tilt = sp.symbols('res epsilon_Tip epsilon_Tilt', 
                                    real=True, positive=True)
    phi_res_tip = sp.symbols( 'phi^res_Tip')
    phi_res_tilt = sp.symbols( 'phi^res_Tilt')
    phi_wind_tip = sp.symbols( 'phi^wind_Tip')
    phi_wind_tilt = sp.symbols( 'phi^wind_Tilt')
    phi_noise_tip = sp.symbols( 'phi^noise_Tip')
    phi_noise_tilt = sp.symbols( 'phi^noise_Tilt')
    
    H_R_tip = sp.symbols( 'H^R_Tip')
    H_N_tip = sp.symbols( 'H^N_Tip')
    H_R_tilt = sp.symbols( 'H^R_Tilt')
    H_N_tilt = sp.symbols( 'H^N_Tilt')
    
    H_R_tipf = sp.Function( 'H^R_Tip^f')(f)
    H_R_tiltf = sp.Function( 'H^R_Tilt^f')(f)
    H_N_tipf = sp.Function( 'H^N_Tip^f')(f)
    H_N_tiltf = sp.Function( 'H^N_Tilt^f')(f)
    H_R_tipz = sp.Function( 'H^R_Tip^z')(z)
    H_R_tiltz = sp.Function( 'H^R_Tilt^z')(z)
    H_N_tipz = sp.Function( 'H^N_Tip^z')(z)
    H_N_tiltz = sp.Function( 'H^N_Tilt^z')(z)
    

    dW_phi = sp.Function('dW_phi')(rho)
    W_phi = sp.Function('W_phi')(rho)

    g_0_tip, g_1_tip = sp.symbols('g^Tip_0 g^Tip_1', real=True)
    g_0_tilt, g_1_tilt = sp.symbols('g^Tilt_0 g^Tilt_1', real=True)
    
    hh, z1, z2 = sp.symbols('h z_1 z_2', positive=True)
    jj, nj, mj, kk, nk, mk = sp.symbols('j n_j m_j k n_k m_k', integer=True)
    R1, R2 = sp.symbols('R_1 R_2', positive=True)

    hMean = sp.symbols('h_mean', positive=True)
    wsMean = sp.symbols('wind_speed_mean', positive=True)
    frHo, fovRadius = sp.symbols('fr_ho fov_radius', positive=True)

    def noisePropagationCoefficient():
        expr0 = ((sp.S.Pi/(180*3600*1000) * D / (4*1e-9)))**2/Nsa
        return sp.Eq(C, expr0)

    def noisePSDTip():
    #    expr0 = noisePropagationCoefficient().rhs * sigma_w / (mu_w**2 * df * DF)
        expr0 = C * sigma_w / (mu_w**2 * df * DF)
        return sp.Eq(phi_noise_tip_f, expr0)

    def noisePSDTilt():
    #    expr0 = noisePropagationCoefficient().rhs * sigma_w / (mu_w**2 * df * DF)
        expr0 = C * sigma_w / (mu_w**2 * df * DF)
        return sp.Eq(phi_noise_tilt_f, expr0)

    # PSD are generated follwing Conan+ 1995, see eq. (8) and (27) 
    def turbPSDTip():
        k0 = 1.0 / L0 # 2 * sp.pi / L0
        with sp.evaluate(False):
            _rhs = 0.0229 * r0**(-sp.S(5) / sp.S(3)) * \
                    (k**2 + k0**2) ** (-sp.S(11) / sp.S(6))
        exprW = _rhs
        expr_k = sp.sqrt((f/V)**2 + k_y**2)
        # Note :
        #   f/(k*V))**2 is cos(mT)^2 see eq (27) Conan+ 1995
        #   order of the bessel is n+1 (n is radial degree)
        #   16 is (n+1)*2*4 with n=1
        expr2 = sp.Integral( 16/(V*sp.pi**2*R**2*k**2)*exprW * (f/(k*V))**2 * 
                            sp.besselj(2, 2*sp.pi*R*k )**2, (k_y, k_y_min, k_y_max))
        expr2 = expr2.subs({k:expr_k})
        return sp.Eq(phi_turb_tip_f, expr2)

    def turbPSDTilt():
        k0 = 1.0 / L0 # 2 * sp.pi / L0
        with sp.evaluate(False):
            _rhs = 0.0229 * r0**(-sp.S(5) / sp.S(3)) * \
                    (k**2 + k0**2) ** (-sp.S(11) / sp.S(6))
        exprW = _rhs
        expr_k = sp.sqrt((f/V)**2 + k_y**2)
        # Note :
        #   (1.0 - (f/(k*V))**2) is sin(mT)^2=1-cos(mT)^2 see eq (27) Conan+ 1995
        #   order of the bessel is n+1 (n is radial degree)
        #   16 is (n+1)*2*4 with n=1
        expr2 = sp.Integral( 16.0/(V*sp.pi**2*R**2*k**2) * exprW * 
                            (1.0 - (f/(k*V))**2) * sp.besselj(2, 2*sp.pi*R*k )**2, 
                            (k_y, k_y_min, k_y_max))
        expr2 = expr2.subs({k:expr_k})
        return sp.Eq(phi_turb_tilt_f, expr2)

    def turbPSDFocus():
        k0 = 1.0 / L0 # 2 * sp.pi / L0
        with sp.evaluate(False):
            _rhs = 0.0229 * r0**(-sp.S(5) / sp.S(3)) * \
                    (k**2 + k0**2) ** (-sp.S(11) / sp.S(6))
        exprW = _rhs
        expr_k = sp.sqrt((f/V)**2 + k_y**2)
        # Note :
        #   f/(k*V))**2 is not present because m (azimuthal frequency of the polynomial)
        #       is 0 see eq (27) Conan+ 1995
        #   order of the bessel is n+1 (n is radial degree)
        #   12 is (n+1)*4 with n=2
        expr2 = sp.Integral( 12/(V*sp.pi**2*R**2*k**2)*exprW * 
                            sp.besselj(3, 2*sp.pi*R*k )**2, (k_y, k_y_min, k_y_max))
        expr2 = expr2.subs({k:expr_k})
        return sp.Eq(phi_turb_focus_f, expr2)
    # end
    
    def sodiumPSDFocus():       
        alphaCoeff = 31.
        betaCoeff = -1.95
        wavelengthInNm = 500
        NaHeight = 90000.
        expr = alphaCoeff * f**betaCoeff
        # focus = delta_height / ( focus_P2V_RMS_ratio * 8. * focal_ratio^2. )
        airmass = 1/sp.cos(zA/180*sp.pi)
        focal_ratio = NaHeight*airmass/(2*R)
        expr2 = expr * ( 1/ (2.*sp.sqrt(3.)*8.*focal_ratio**2.) * 1e9 )**2.
        
        return sp.Eq(phi_sodium_focus_f, expr2)
    
    def residualTT():
        return sp.Eq(res, sp.sqrt(e_tip+e_tilt))

    def residualTip():
        return sp.Eq(e_tip, sp.Integral(phi_res_tip, (f, f_min, f_max)) )

    def residualTilt():
        return sp.Eq(e_tilt, sp.Integral(phi_res_tilt, (f, f_min, f_max)) )

    def residualTipPSD():
        return sp.Eq(phi_res_tip, sp.Abs(H_R_tip)**2 * phi_wind_tip + 
                     sp.Abs(H_N_tip)**2 * phi_noise_tip)

    def residualTiltPSD():
        return sp.Eq(phi_res_tilt, sp.Abs(H_R_tilt)**2 * phi_wind_tilt + 
                     sp.Abs(H_N_tilt)**2 * phi_noise_tilt)

    # 4 tf in z with 1 gain each to tune
    def ztfTipWindMono():
        return sp.Eq(H_R_tipz, (1-z**-1)/(1-z**-1+g_0_tip*z**-d))

    def ztfTiltWindMono():
        return sp.Eq(H_R_tiltz, (1-z**-1)/(1-z**-1+g_0_tilt*z**-d))

    def ztfTipNoiseMono():
        return sp.Eq(H_N_tipz, g_0_tip*z**-d/(1-z**-1+g_0_tip*z**-d))

    def ztfTiltNoiseMono():
        return sp.Eq(H_N_tiltz, g_0_tilt*z**-d/(1-z**-1+g_0_tilt*z**-d))
    # end

    # 4 tf in z with 2 gains each to tune
    def ztfTipWind():
        return sp.Eq( H_R_tipz, (1-1.9995*z**-1+0.9995*z**-2)/ \
                     ((1-1.9995*z**-1+0.9995*z**-2)+g_0_tip*g_1_tip*(1-1.3*z**-1+0.3825*z**-2)*(z**-d)) )

    def ztfTiltWind():
        return sp.Eq( H_R_tiltz, (1-1.9995*z**-1+0.9995*z**-2)/ \
                     ((1-1.9995*z**-1+0.9995*z**-2)+g_0_tilt*g_1_tilt*(1-1.3*z**-1+0.3825*z**-2)*(z**-d)) )

    def ztfTipNoise():
        return sp.Eq( H_N_tipz, (g_0_tip*g_1_tip*(1-1.3*z**-1+0.3825*z**-2)*(z**-d))/ \
                     ((1-1.9995*z**-1+0.9995*z**-2)+g_0_tip*g_1_tip*(1-1.3*z**-1+0.3825*z**-2)*(z**-d)) )

    def ztfTiltNoise():
        return sp.Eq( H_N_tiltz, (g_0_tilt*g_1_tilt*(1-1.3*z**-1+0.3825*z**-2)*(z**-d))/ \
                     ((1-1.9995*z**-1+0.9995*z**-2)+g_0_tilt*g_1_tilt*(1-1.3*z**-1+0.3825*z**-2)*(z**-d)) )
    # end

    # 4 tf in f obtained from corresponding tf in z
    def tfTipWind(ztf = ztfTipWind()):
        return sp.Eq(H_R_tipf, 
                     subsParamsByName(ztf.rhs, {'z':sp.exp(2*sp.pi*f*sp.I/f_loop)}))
    
    def tfTiltWind(ztf = ztfTiltWind()):
        return sp.Eq(H_R_tiltf, 
                     subsParamsByName(ztf.rhs, {'z':sp.exp(2*sp.pi*f*sp.I/f_loop)}))

    def tfTipNoise(ztf = ztfTipNoise()):
        return sp.Eq(H_N_tipf, 
                     subsParamsByName(ztf.rhs, {'z':sp.exp(2*sp.pi*f*sp.I/f_loop)}))

    def tfTiltNoise(ztf = ztfTiltNoise()):
        return sp.Eq(H_N_tiltf, 
                     subsParamsByName(ztf.rhs, {'z':sp.exp(2*sp.pi*f*sp.I/f_loop)}))
    # end
    
    
    # 4 tf in f obtained from corresponding tf in z
    def tfTipWindMono(ztf = ztfTipWindMono()):
        return sp.Eq(H_R_tipf, 
                     subsParamsByName(ztf.rhs, {'z':sp.exp(2*sp.pi*f*sp.I/f_loop)}))
    
    def tfTiltWindMono(ztf = ztfTiltWindMono()):
        return sp.Eq(H_R_tiltf, 
                     subsParamsByName(ztf.rhs, {'z':sp.exp(2*sp.pi*f*sp.I/f_loop)}))

    def tfTipNoiseMono(ztf = ztfTipNoiseMono()):
        return sp.Eq(H_N_tipf, 
                     subsParamsByName(ztf.rhs, {'z':sp.exp(2*sp.pi*f*sp.I/f_loop)}))

    def tfTiltNoiseMono(ztf = ztfTiltNoiseMono()):
        return sp.Eq(H_N_tiltf, 
                     subsParamsByName(ztf.rhs, {'z':sp.exp(2*sp.pi*f*sp.I/f_loop)}))
    # end

    def interactionMatrixNGS():
        expr0 = D + 2 * H_DM * r_FoV
        expr1 = 2 * H_DM * D / DP**2
        P = sp.Matrix([[1,0,C*2*sp.S(sp.sqrt(3))*x_NGS, C*sp.S(sp.sqrt(6))*y_NGS, C*sp.S(sp.sqrt(6))*x_NGS], 
                        [0,1,C*2*sp.S(sp.sqrt(3))*y_NGS, C*sp.S(sp.sqrt(6))*x_NGS, -C*sp.S(sp.sqrt(6))*y_NGS] ])
        P = P.subs({C:expr1})
        P = P.subs({DP:expr0})
        return P

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

    def completeIntegralTipLOandTf():
        ztfW = ztfTipWindMono()
        ztfN = ztfTipNoiseMono()
        tfW = tfTipWind(ztfW).rhs
        tfN = tfTipNoise(ztfN).rhs
        completeIntegralTipV = subsParamsByName( residualTip().rhs, {'phi^res_Tip':residualTipPSD().rhs} )
        completeIntegralTipV = subsParamsByName( completeIntegralTipV, {'H^R_Tip':tfW, 'H^N_Tip':tfN} )
        return completeIntegralTipV, tfW, tfN, ztfW.rhs, ztfN.rhs

    def completeIntegralTiltLOandTf():
        ztfW = ztfTiltWindMono()
        ztfN = ztfTiltNoiseMono()
        tfW = tfTiltWind(ztfW).rhs
        tfN = tfTiltNoise(ztfN).rhs
        completeIntegralTiltV = subsParamsByName( residualTilt().rhs, {'phi^res_Tilt':residualTiltPSD().rhs} )
        completeIntegralTiltV = subsParamsByName( completeIntegralTiltV, {'H^R_Tilt':tfW, 'H^N_Tilt':tfN} )
        return completeIntegralTiltV, tfW, tfN, ztfW.rhs, ztfN.rhs
    
    def completeIntegralTipAndTf():
        ztfW = ztfTipWind()
        ztfN = ztfTipNoise()
        tfW = tfTipWind(ztfW).rhs
        tfN = tfTipNoise(ztfN).rhs
        completeIntegralTipV = subsParamsByName( residualTip().rhs, {'phi^res_Tip':residualTipPSD().rhs} )
        completeIntegralTipV = subsParamsByName( completeIntegralTipV, {'H^R_Tip':tfW, 'H^N_Tip':tfN} )
        return completeIntegralTipV, tfW, tfN, ztfW.rhs, ztfN.rhs
    
    def completeIntegralTiltAndTf():
        ztfW = ztfTiltWind()
        ztfN = ztfTiltNoise()
        tfW = tfTiltWind(ztfW).rhs
        tfN = tfTiltNoise(ztfN).rhs
        completeIntegralTiltV = subsParamsByName( residualTilt().rhs, {'phi^res_Tilt':residualTiltPSD().rhs} )
        completeIntegralTiltV = subsParamsByName( completeIntegralTiltV, {'H^R_Tilt':tfW, 'H^N_Tilt':tfN} )
        return completeIntegralTiltV, tfW, tfN, ztfW.rhs, ztfN.rhs
    
    def zernikeCovarianceI():
        _integrand = zernikeCovarianceD().rhs
        _rhs = sp.Integral(_integrand, (f, f_min, f_max))
        return sp.Eq(W_phi, _rhs)
    
    def zernikeCovarianceD():
        f0 = (-1)**mk * sp.sqrt((nj+1)*(nk+1)) * sp.I**(nj+nk) * 2 ** ( 1 - 0.5*((sp.KroneckerDelta(0,mj) + sp.KroneckerDelta(0,mk))) )
        f1 = 1 / (sp.pi * R1 * R2) 
        ff0 = 1 / L0
        with sp.evaluate(False):
            psd_def = 0.0229*r0**(-sp.S(5)/sp.S(3))*(f**2+ff0**2)**(-sp.S(11)/sp.S(6))
        f3 = sp.cos( (mj+mk)*theta + (sp.pi/4) * ( (1-sp.KroneckerDelta(0, mj)) * ((-1)**jj-1) + (1-sp.KroneckerDelta(0, mk)) * ((-1)**kk-1)) )
        f4 = sp.I**(3*(mj+mk)) * sp.besselj( (mj+mk), 2*sp.pi*f*hh*rho)
        f5 = sp.cos( (mj-mk)*theta + (sp.pi/4) * ( (1-sp.KroneckerDelta(0, mj)) * ((-1)**jj-1) - (1-sp.KroneckerDelta(0, mk)) * ((-1)**kk-1)) )
        f6 = sp.I**(3*sp.Abs(mj-mk)) * sp.besselj( sp.Abs(mj-mk), 2*sp.pi*f*hh*rho)    
        _rhs = f0 * f1 * (psd_def * sp.besselj( nj+sp.Integer(1), 2*sp.pi*f*R1) * sp.besselj( nj+sp.Integer(1), 2*sp.pi*f*R2) / f) * (f3*f4+f5*f6)    
        return sp.Eq(dW_phi, _rhs)

    def zernikeCovarianceDfiltered():
        f0 = (-1)**mk * sp.sqrt((nj+1)*(nk+1)) * sp.I**(nj+nk) * 2 ** ( 1 - 0.5*((sp.KroneckerDelta(0,mj) + sp.KroneckerDelta(0,mk))) )
        f1 = 1 / (sp.pi * R1 * R2)
        ff0 = 1 / L0

        # meta-pupil radius
        arcsec2rad = sp.S.Pi/(180*3600)
        # we use average values
        cur_R1 = R1 + hMean * sp.tan(fovRadius*arcsec2rad)
        f2p = 2*sp.S.Pi*f

        # ------------------------------------------------------------------------------------------------------
        #    original
        # ------------------------------------------------------------------------------------------------------
        # this filter removes up to the 2nd radial order in the turbulence covariance (see filters in Sasiela93)
        #filt = 1 -   (2*sp.besselj(1,f2p*cur_R1)/(f2p*cur_R1))**2 \
        #         -   (4*sp.besselj(2,f2p*cur_R1)/(f2p*cur_R1))**2 \
        #         - 9*(2*sp.besselj(3,f2p*cur_R1)/(f2p*cur_R1))**2
        #
        # -----> original cur_rtf
        # equivalence spatial frequency <=> temporal frequency
        # HO loop RTF computation (frHo is HO loop framerate in Hz)
        #zVar = sp.exp(2*sp.S.Pi*sp.I*ws*f/frHo)
        #cVar = 0.25 * zVar**(-3)
        #OLTF = cVar/(1-zVar**(-1))
        #cur_rtf = (1/(1+OLTF))**2
        #
        # -----> approximation of cur_rtf
        #cur_rtf = f**2/(f**2+frHo/ws**2)
        #
        # correction so that the RTF does not reject first 2 radial orders (TTFA) and global piston
        #rtf = 1 - (1-cur_rtf)*filt
        # ------------------------------------------------------------------------------------------------------
        #    new
        # ------------------------------------------------------------------------------------------------------
        filt = 1 - ( (2*sp.besselj(1,f2p*cur_R1))**2 \
                 +   (4*sp.besselj(2,f2p*cur_R1))**2 \
                 + 9*(2*sp.besselj(3,f2p*cur_R1))**2 )/(f2p*cur_R1)**2
        lpf = 0.8*frHo/wsMean**2/(f**2+0.8*frHo/wsMean**2)
        rtf = 1 - lpf*filt

        with sp.evaluate(False):
            psd_def = 0.0229*r0**(-sp.S(5)/sp.S(3))*(f**2+ff0**2)**(-sp.S(11)/sp.S(6))
        f3 = sp.cos( (mj+mk)*theta + (sp.pi/4) * ( (1-sp.KroneckerDelta(0, mj)) * ((-1)**jj-1) + (1-sp.KroneckerDelta(0, mk)) * ((-1)**kk-1)) )
        f4 = sp.I**(3*(mj+mk)) * sp.besselj( (mj+mk), 2*sp.pi*f*hh*rho)
        f5 = sp.cos( (mj-mk)*theta + (sp.pi/4) * ( (1-sp.KroneckerDelta(0, mj)) * ((-1)**jj-1) - (1-sp.KroneckerDelta(0, mk)) * ((-1)**kk-1)) )
        f6 = sp.I**(3*sp.Abs(mj-mk)) * sp.besselj( sp.Abs(mj-mk), 2*sp.pi*f*hh*rho)
        _rhs = f0 * f1 * (psd_def * rtf * sp.besselj( nj+sp.Integer(1), 2*sp.pi*f*R1) * sp.besselj( nj+sp.Integer(1), 2*sp.pi*f*R2) / f) * (f3*f4+f5*f6)
        return sp.Eq(dW_phi, _rhs)

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

    def gaussianMean():
        f_k = sp.symbols('f_k', real=True)
        sigma_ron, b, t, nu, F = sp.symbols('sigma_RON b t nu F', real=True)
        sigma_e = sp.sqrt(F*(f_k+b)+sigma_ron*sigma_ron)
        f1 = subsParamsByName( expr_phi(), {'x': (t-f_k)/sigma_e})
        f2 = subsParamsByName( expr_Phi(), {'x': (f_k-t)/sigma_e})
        f3 = subsParamsByName( expr_Phi(), {'x': (t-f_k)/sigma_e})
        _rhs = sigma_e * f1  + f_k * f2 + nu * f3
        _lhs = sp.symbols('mu_k_thr')
        return _rhs

    def gaussianVariance():
        mu_k = sp.symbols('mu_k_thr')
        f_k = sp.symbols('f_k', real=True)
        sigma_ron, b, t, nu, F = sp.symbols('sigma_RON b t nu F', real=True)
        sigma_e = sp.sqrt(F*(f_k+b)+sigma_ron*sigma_ron)
        f1 = subsParamsByName( expr_phi(), {'x': (t-f_k)/sigma_e})
        f2 = subsParamsByName( expr_Phi(), {'x': (f_k-t)/sigma_e})
        f3 = subsParamsByName( expr_Phi(), {'x': (t-f_k)/sigma_e})
        _rhs = sigma_e*(t+f_k) * f1  + (f_k*f_k + sigma_e*sigma_e) * f2 + nu**2 * f3
        _lhs = sp.symbols('sigma^2_k_thr')
        return _rhs

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
        _lhs = sp.symbols('mu_k_thr')
        return sp.Eq(_lhs, _rhs)

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
        _lhs = sp.symbols('sigma^2_k_thr')
        return sp.Eq(_lhs, _rhs)


    def truncatedMeanIntegrand():
        F, z = sp.symbols('F z', real=True)
        i = sp.symbols('i', integer=True)
        f_k = sp.symbols('f_k', real=True)
        sigma_ron, b, t, nu = sp.symbols('sigma_RON b t nu', real=True)
        f4 = subsParamsByName(expr_phi(), {'x': (t-(z-b))/sigma_ron})
        f5 = subsParamsByName(expr_Phi(), {'x': (z-b-t)/sigma_ron})
        f6 = subsParamsByName(expr_Phi(), {'x': (t-(z-b))/sigma_ron})
        _rhs = expr_G() * ( sigma_ron * f4 + (z-b) * f5 + nu * f6 )
        _lhs = sp.symbols('I_mu_k_thr')
        return sp.Eq(_lhs, _rhs)


    def truncatedVarianceIntegrand():
        F, z = sp.symbols('F z', real=True)
        i = sp.symbols('i', integer=True)
        f_k = sp.symbols('f_k', real=True)
        sigma_ron, b, t, nu = sp.symbols('sigma_RON b t nu', real=True)
        f4 = subsParamsByName(expr_phi(), {'x': (t-(z-b))/sigma_ron})
        f5 = subsParamsByName(expr_Phi(), {'x': (z-b-t)/sigma_ron})
        f6 = subsParamsByName(expr_Phi(), {'x': (t-(z-b))/sigma_ron})
        _rhs = expr_G() * ( sigma_ron * (t+z-b) * f4 + (sigma_ron**2 + (z-b)**2) * f5 + nu**2 * f6 )
        _lhs = sp.symbols('I_sigma_k_thr')
        return sp.Eq(_lhs, _rhs)


    def truncatedMeanComponents(ii):
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
        _integrand = subsParamsByName(truncatedMeanIntegrand().rhs, {'z':z, 'i':i} )        
        if ii==0:
            return expr10
        elif ii==1:
            return expr_K_i
        elif ii==2:
            return sp.Integral( _integrand, (z, 0.0001, z_max))

    def truncatedMean():
        expr10, expr_K_i, integral =  truncatedMeanComponents(0), truncatedMeanComponents(1), truncatedMeanComponents(2)
        i_max = sp.symbols('i_max', integer=True, positive=True)
        i = getSymbolByName(expr_K_i, 'i')
        _rhs = expr10 + sp.Sum( expr_K_i *  integral , (i, 1, i_max) )
        _lhs = sp.symbols('mu_k_thr')
        return sp.Eq(_lhs, _rhs)


    def truncatedVarianceComponents(ii):
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
        _integrand = subsParamsByName(truncatedVarianceIntegrand().rhs, {'z':z, 'i':i} )
        if ii==0:
            return expr20
        elif ii==1:
            return expr_K_i
        elif ii==2:
            return sp.Integral( _integrand, (z, 0.0001, z_max))


    def truncatedVariance():
        expr20, expr_K_i, integral =  truncatedVarianceComponents(0), truncatedVarianceComponents(1), truncatedVarianceComponents(2)
        i_max = sp.symbols('i_max', integer=True, positive=True)
        mu_k = sp.symbols('mu_k_thr')
        i = getSymbolByName(expr_K_i, 'i')
        _rhs = expr20 + sp.Sum(expr_K_i*integral , (i, 1, i_max)) - mu_k**2
        _lhs = sp.symbols('sigma^2_k_thr')
        return sp.Eq(_lhs, _rhs)


    _MavisFormulas = Formulary("MAVIS",
                            ['noisePropagationCoefficient',
                             'noisePSDTip',
                             'noisePSDTilt',
                             'turbPSDTip',
                             'turbPSDTilt',
                             'turbPSDFocus',
                             'sodiumPSDFocus',
                             'interactionMatrixNGS',                                
                             'residualTT',
                             'residualTip',
                             'residualTilt',
                             'residualTipPSD',
                             'residualTiltPSD',
                             'ztfTipWindMono',
                             'ztfTiltWindMono',
                             'ztfTipNoiseMono',
                             'ztfTiltNoiseMono',
                             'ztfTipWind',
                             'ztfTiltWind',
                             'ztfTipNoise',
                             'ztfTiltNoise',
                             'tfTipWind',
                             'tfTiltWind',
                             'tfTipNoise',
                             'tfTiltNoise',
                             'tfTipWindMono',
                             'tfTiltWindMono',
                             'tfTipNoiseMono',
                             'tfTiltNoiseMono',
                             'completeIntegralTipLO',
                             'completeIntegralTiltLO',
                             'completeIntegralTip',
                             'completeIntegralTilt',
                             'completeIntegralTipLOandTf',
                             'completeIntegralTiltLOandTf',
                             'completeIntegralTipAndTf',
                             'completeIntegralTiltAndTf',
                             'ZernikeCovarianceD',
                             'ZernikeCovarianceDfiltered',
                             'ZernikeCovarianceI',
                             'GaussianMean',
                             'GaussianVariance',
                             'TruncatedMeanBasic',
                             'TruncatedVarianceBasic',
                             'TruncatedMean',
                             'TruncatedMeanIntegrand',
                             'TruncatedVariance',
                             'TruncatedVarianceIntegrand',
                             'truncatedMeanComponents0',
                             'truncatedMeanComponents1',
                             'truncatedMeanComponents2',
                             'truncatedVarianceComponents0',
                             'truncatedVarianceComponents1',
                             'truncatedVarianceComponents2'
                            ],
                            [
                             noisePropagationCoefficient(),
                             noisePSDTip(),
                             noisePSDTilt(),
                             turbPSDTip(),
                             turbPSDTilt(),
                             turbPSDFocus(),
                             sodiumPSDFocus(),
                             interactionMatrixNGS(),                                
                             residualTT(),
                             residualTip(),
                             residualTilt(),
                             residualTipPSD(),
                             residualTiltPSD(),
                             ztfTipWindMono(),
                             ztfTiltWindMono(),
                             ztfTipNoiseMono(),
                             ztfTiltNoiseMono(),
                             ztfTipWind(),
                             ztfTiltWind(),
                             ztfTipNoise(),
                             ztfTiltNoise(),
                             tfTipWind(),
                             tfTiltWind(),
                             tfTipNoise(),
                             tfTiltNoise(),
                             tfTipWindMono(),
                             tfTiltWindMono(),
                             tfTipNoiseMono(),
                             tfTiltNoiseMono(),
                             completeIntegralTipLO(),
                             completeIntegralTiltLO(),
                             completeIntegralTip(),
                             completeIntegralTilt(),
                             completeIntegralTipLOandTf(),
                             completeIntegralTiltLOandTf(),
                             completeIntegralTipAndTf(),
                             completeIntegralTiltAndTf(),
                             zernikeCovarianceD(),
                             zernikeCovarianceDfiltered(),
                             zernikeCovarianceI(),
                             gaussianMean(),
                             gaussianVariance(),
                             truncatedMeanBasic(), 
                             truncatedVarianceBasic(),
                             truncatedMean(), 
                             truncatedMeanIntegrand(), 
                             truncatedVariance(),
                             truncatedVarianceIntegrand(),
                             truncatedMeanComponents(0),
                             truncatedMeanComponents(1),
                             truncatedMeanComponents(2),
                             truncatedVarianceComponents(0),
                             truncatedVarianceComponents(1),
                             truncatedVarianceComponents(2)
                           ] )

    return _MavisFormulas

_mavisFormulas = createMavisFormulary()
