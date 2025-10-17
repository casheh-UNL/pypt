# Classes and functions for calculating GW spectra from FOPT, PBH, etc.

from .constants import *
from .bubble_nucleation import *
from .eff_DoF.eff_DoF import g_SM

from scipy.optimize import minimize_scalar

class GravitationalWaves:
    def __init__(self, Veff: VeffBL, T_PT=-1, T_c=-1, alpha=None, beta_over_H=None, v_wall=None, kappa_v=None, kappa_turb=None, kappa_col=None, supercooling=False):
        
        self.Veff = Veff
        self.FOPT = BubbleNucleationBL(Veff, T_PT=T_PT, T_c=T_c)
        self.T_PT = T_PT if T_PT != -1 else self.FOPT.TPT
        self.alpha = alpha if alpha is not None else abs(self.FOPT.alpha())
        self.beta_over_H = beta_over_H if beta_over_H is not None else self.FOPT.betaByHstar()

        self.T_GW = self.T_PT * (1 + self.alpha)**(1/4) # GWs are produced at the reheating temperature

        self.g_eff_GW = g_SM(self.T_GW) + 2*1 + 1*3 + 1*1 + 1*1 # https://arxiv.org/pdf/1609.04979.pdf

        self.v_wall = v_wall if v_wall is not None else self.find_v_wall_thermal_eq()

        self.kappa_v = kappa_v if kappa_v is not None else self.find_kappa_v(self.v_wall, self.alpha)
        self.kappa_turb = kappa_turb if kappa_turb is not None else 0.05*self.kappa_v
        self.kappa_col = kappa_col if kappa_col is not None else 1 - self.kappa_v - self.kappa_turb

        self.supercooling = supercooling

    # wall speed as a function of alpha; https://arxiv.org/pdf/2303.10171.pdf eq. (34)
    def find_v_wall_enthalpy_ratio(self):
        alpha = self.alpha
        cs2 = 1/3 # sound speed in the broken phase...IT IS ASSUMED THAT SOUND SPEED IS CONSTANT AND EQUAL IN BOTH PHASES
        a, b, p = 0.2233, 1.704, -3.433 # fitted parameters

        # pressure p is set equal to -Veff / eq. (1)
        def p_symmetric(T):
            phi_symmetric = self.Veff.phi_min
            return -self.Veff.Veff_no_shift(phi_symmetric, T)
        def p_broken(T):
            phi_broken = self.Veff.find_minimum_phi(T)
            return -self.Veff.Veff_no_shift(phi_broken, T)
        
        # w = T * dp/dT = e + p / eq. (2)
        def w_symmetric(T):
            return T * (p_symmetric(1.001*T) - p_symmetric(0.999*T)) / (1.001 * T - 0.999 * T)
        def w_broken(T):
            return T * (p_broken(1.001*T) - p_broken(0.999*T)) / (1.001 * T - 0.999 * T)
        
        T_n = self.T_PT # nucleation temperature
        Psi_n = float( w_broken(T_n) / w_symmetric(T_n) ) # ratio of enthalpies in the broken and symmetric phase / eq. (19)
        v_low = ( (3*alpha + Psi_n - 1) / 2 / (2 - 3*Psi_n + Psi_n**3) )**0.5 # accurate for v_wall <~ 0.5
        v_J = cs2**(0.5) * ( (1 + (3*alpha*(1 - cs2 + 3*cs2*alpha))**(0.5)) / (1 + 3*cs2*alpha) ) # Jouguet speed
        v_high = v_J * (1 - a*(1 - Psi_n)**b / alpha) # complex-valued for Psi_n > 1

        v_fit = (abs(v_low)**p + abs(v_high)**p)**(1/p)
        return v_fit

    # wall speed in thermal equilibrium; https://arxiv.org/pdf/2111.02393.pdf eq. (8.1)
    def find_v_wall_thermal_eq(self):
        TGW, alpha = self.T_GW, self.alpha
        if alpha < 0:
            return None
        cs2 = 1/3 # sound speed in the broken phase
        v_J = cs2**(0.5) * ( (1 + (3*alpha*(1 - cs2 + 3*cs2*alpha))**(0.5)) / (1 + 3*cs2*alpha) ) # Jouguet speed
        phi_min = self.Veff.find_minimum_phi(TGW)
        DeltaV = -self.Veff(phi_min, TGW)
        rho_rad = pi**2 * self.g_eff_GW * TGW**4 / 30
        v = (DeltaV / alpha / rho_rad)**0.5
        if v < v_J:
            return v
        elif v >= v_J:
            return 1.0

    # numerical fit for kappa_v: the fraction of latent heat transferred into the bulk motion of the plasma
    # https://arxiv.org/pdf/1004.4187.pdf
    #! divide by zero error for high alpha...needed to implement a limit for vw = 1
    def find_kappa_v(self, v_wall, alpha):
        cs2 = 1/3
        cs = cs2**(0.5)

        def xi_J(alpha_N):
            return ((2*alpha_N/3 + alpha_N**2)**(0.5) + 1/3**(0.5)) / (1 + alpha_N)

        def kappa_A(xi_w, alpha_N):
            return xi_w**(6/5) * 6.9*alpha_N / (1.36 - 0.037*alpha_N**(0.5) + alpha_N)

        def kappa_B(alpha_N):
            return alpha_N**(2/5) / (0.017 + (0.997 + alpha_N)**(2/5))

        def kappa_C(alpha_N):
            return alpha_N**(0.5) / (0.135 + (0.98 + alpha_N)**(0.5))

        def kappa_D(alpha_N):
            return alpha_N / (0.73 + 0.083*alpha_N**(0.5) + alpha_N)

        def delta_kappa(alpha_N):
            return -0.9 * np.log(alpha_N**(0.5) / (1 + alpha_N**(0.5)))

        # subsonic deflagrations (xi_w <~ cs)
        def kappa_def(xi_w, alpha_N):
            num = cs**(11/5) * kappa_A(xi_w, alpha_N) * kappa_B(alpha_N)
            den = (cs**(11/5) - xi_w**(11/5))*kappa_B(alpha_N) + xi_w*cs**(6/5) * kappa_A(xi_w, alpha_N)
            return num / den

        # detonations (xi_w >~ xi_J)
        def kappa_det(xi_w, alpha_N):
            num = (xi_J(alpha_N) - 1)**3 * xi_J(alpha_N)**(5/2) * xi_w**(-5/2) * kappa_C(alpha_N) * kappa_D(alpha_N)
            den = ((xi_J(alpha_N) - 1)**3 - (xi_w - 1)**3) * xi_J(alpha_N)**(5/2) * kappa_C(alpha_N) + (xi_w - 1)**3 * kappa_D(alpha_N)
            return num / den

        # hybrid/supersonic deflagration (cs < xi_w < xi_J)
        def kappa_hyb(xi_w, alpha_N):
            return kappa_B(alpha_N) + (xi_w - cs)*delta_kappa(alpha_N) + \
            (xi_w - cs)**3 / (xi_J(alpha_N) - cs)**3 * (kappa_C(alpha_N) - kappa_B(alpha_N) - (xi_J(alpha_N) - cs)*delta_kappa(alpha_N))
        
        # blend the separate kappas at v_w = cs and v_w = xi_J

        # sigmoid blending function
        def B(x, x0, transition_width):
            return 1 / (1 + np.exp(-(x - x0) / transition_width))

        # blended kappa
        def kappa(v_w, alpha_N, transition_width=1e-3):
            def def_to_hyb(v_w, alpha_N):
                return (1 - B(v_w, cs, transition_width)) * kappa_def(v_w, alpha_N) + B(v_w, cs, transition_width) * kappa_hyb(v_w, alpha_N)
            
            return (1 - B(v_w, xi_J(alpha_N), transition_width)) * def_to_hyb(v_w, alpha_N) + B(v_w, xi_J(alpha_N), transition_width) * kappa_det(v_w, alpha_N)
        
        if alpha is None:
            # limiting approximations for α >> 1:
            # κΑ = 6.9 vw^(6/5), κB = κC = κD = 1
            # ξJ = 1, δκ = 0
            xi_Jo = 1
            kappa_defl = cs**(11/5)*6.9*v_wall**(6/5) \
                        / ( (cs**(11/5) - v_wall**(11/5)) + 6.9*cs**(6/5)*v_wall**(11/5) )
            kappa_deto = 0
            kappa_hybr = 1
            def kappa(v_w, transition_width=1e-3):
                def def_to_hyb(v_w):
                    return (1 - B(v_w, cs, transition_width)) * kappa_defl + B(v_w, cs, transition_width) * kappa_hybr
                return (1 - B(v_w, xi_Jo, transition_width)) * def_to_hyb(v_w) + B(v_w, xi_Jo, transition_width) * kappa_deto
            return kappa(v_wall)
        
        elif v_wall < 1:
            return kappa(v_wall, alpha)
        else:
            return kappa_D(alpha)
    
    def alpha_term(self, alpha):
        # In the limit of infinite alpha, the alpha dependence in the
        # GW signal drops out.
        if alpha == None:
            # limit of infinite alpha
            return 1
        else:
            return alpha / (1+alpha)
        
    def Htau_sh(self):
        """
        Shock time or sound wave lifetime.
        https://arxiv.org/pdf/2007.08537
        https://arxiv.org/pdf/2003.07360
        """
        alpha = self.alpha
        vw = self.v_wall
        beta_over_H = self.beta_over_H

        U_f = np.sqrt(3/4 * self.kappa_v * self.alpha_term(alpha))
        return (8*np.pi)**(1/3) * vw / beta_over_H / U_f
    def Htau_sw(self):
        return min(1, self.Htau_sh())
        
    ###################################
    # SOUND WAVE/ACOUSTIC CONTRIBUTIONS
    ###################################
    def f_sw_peak(self):
        return 1.9e-5 * (1/self.v_wall) * (self.beta_over_H) * (self.T_GW / 100 / GeV) * (self.g_eff_GW / 100)**(1/6)
    
    def Omega_sw_h2(self, f):
        g_eff_PT = self.g_eff_GW
        v_w = self.v_wall
        beta_over_H = self.beta_over_H
        alpha = self.alpha
        kappa = self.kappa_v
        f_peak = self.f_sw_peak()

        result = 2.62e-6 * v_w * (1/beta_over_H) * (kappa * self.alpha_term(alpha))**2 * (g_eff_PT/100)**(-1/3) * \
        7**3.5 * (f / f_peak)**3 / (4 + 3*(f / f_peak)**2)**3.5

        result *= self.Htau_sw() # suppression factor from sound wave lifetime

        # Peisi found an erratum in this paper: https://arxiv.org/pdf/1704.05871.pdf.
        # They missed a factor of 3.
        return result * 3
    
    ################################
    # BUBBLE COLLISION CONTRIBUTIONS
    ################################
    # for supercooled FOPTs, see https://arxiv.org/pdf/2007.04967.pdf (page 5)
    def f_col_peak(self):
        if self.supercooling:
            return 1.65e-5 * (self.T_GW / 100 / GeV) * (self.g_eff_GW / 100)**(1/6) * self.beta_over_H * 0.81 / (2*np.pi)
        else:
            return 1.65e-5 * (0.62 / (1.8 - 0.1*self.v_wall + self.v_wall**2)) * self.beta_over_H * (self.T_GW / 100 / GeV) * (self.g_eff_GW / 100)**(1/6)

    def Omega_col_h2(self, f):
        g_eff_PT = self.g_eff_GW
        v_w = self.v_wall
        beta_over_H = self.beta_over_H
        alpha = self.alpha
        kappa = self.kappa_col
        f_peak = self.f_col_peak()
        
        if self.supercooling:
            def S(f):
                hbar = 6.582119569e-25 # GeV s
                GeV_to_Hz = 1/hbar

                omega = f * 2*np.pi
                A = 3.63/100
                # omega_bar = 0.81 * self.beta*GeV_to_Hz
                omega_d = 0.13 * self.beta*GeV_to_Hz
                a, b, c, d = 2.54, 2.24, 2.30, 0.93

                omega_bar_today = self.f_col_peak() * 2*np.pi
                term1 = (1 + np.power(omega/omega_d, d-a)) / (1 + np.power(omega_bar_today/omega_d, d-a))
                term2 = A * np.power(a+b, c) / np.power(b * np.power(omega/omega_bar_today, -a/c) + a * np.power(omega/omega_bar_today, b/c), c)
                return term1 * term2
            
            return 1.67e-5 * beta_over_H**(-2) * (kappa * self.alpha_term(alpha))**2 * (g_eff_PT / 100)**(-1/3) * S(f)
        else:
            return 1.67e-5 * (0.11*v_w**3 / (0.42 + v_w**2)) * beta_over_H**(-2) * (kappa * self.alpha_term(alpha))**2 * (g_eff_PT / 100)**(-1/3) * \
            3.8*(f / f_peak)**2.8 / (1 + 0.28*(f / f_peak)**3.8)
    
    ##############################################
    # MAGNETOHYDRODYNAMIC TURBULENCE CONTRIBUTIONS
    ##############################################
    def h_star(self):
        return 1.65e-5 * (self.T_GW / 100 / GeV) * (self.g_eff_GW / 100)**(1/6)

    def f_turb(self):
        return 2.7e-5 * (1 / self.v_wall) * self.beta_over_H * (self.T_GW / 100 / GeV) * (self.g_eff_GW / 100)**(1/6)
    
    def Omega_turb_h2(self, f):
        g_eff_PT = self.g_eff_GW
        v_w = self.v_wall
        beta_over_H = self.beta_over_H
        alpha = self.alpha
        kappa = self.kappa_v # self.kappa_turb
        f_star = self.f_turb()

        h_star = self.h_star()

        return 3.35e-4 * v_w * beta_over_H**(-1) * (kappa * self.alpha_term(alpha))**(3/2) * (g_eff_PT / 100)**(-1/3) * \
        (f / f_star)**3 / (1 + f / f_star)**(11/3) / (1 + 8 * np.pi * f / h_star) \
        * (1 - self.Htau_sw()) # sound waves dissipating into turbulence?
    
    def f_turb_peak(self):
        # here is where the turbulence spectrum peaks after setting its first derivative equal to zero
        f_turb = self.f_turb()
        h_star = self.h_star()

        return (1/40/np.pi) * (-h_star + 24*np.pi*f_turb + np.sqrt(h_star**2 + 312*np.pi*h_star*f_turb + 576*np.pi**2 *f_turb**2))
    
    ########################################################################################################################
    # TOTAL (assuming a linear combination)
    def Omega_h2(self, f):
        return self.Omega_sw_h2(f) + self.Omega_col_h2(f) + self.Omega_turb_h2(f)
    
    def find_peak(self, contribution='total'):
        # Returns a tuple (f_peak, Omega_h2_peak)
        if contribution == 'total':
            def Omega_h2(f): return self.Omega_h2(f)
        elif contribution == 'sw':
            def Omega_h2(f): return self.Omega_sw_h2(f)
        elif contribution == 'bc':
            def Omega_h2(f): return self.Omega_col_h2(f)
        elif contribution == 'mhd':
            def Omega_h2(f): return self.Omega_turb_h2(f)

        f_peaks = (self.f_col_peak(), self.f_sw_peak(), self.f_turb_peak())
        f_min, f_max = (min(f_peaks), max(f_peaks))
        bracket = [1e-3*f_min, 1*f_min, 1e3*f_max]
        # problems with turbulence:
        # if contribution == 'mhd':
        #     f = self.f_turb_peak()
        #     bracket = [1e-3*f, 1*f, 1e3*f]
        #     print(f'f_turb_peak = {f}\n',
        #           f'bracket = {bracket}\n',
        #           f'-Omega_turb_h2(bracket) = {(-Omega_h2(bracket[0]), -Omega_h2(bracket[1]), -Omega_h2(bracket[2]))}')
        if Omega_h2(bracket[1]) == 0: # happens with turbulence with (1-Hτ) factor
            return (None, None)
        else:
            try:
                result = minimize_scalar(lambda f: -Omega_h2(f), bracket=bracket)
                if result.success is True:
                    return (result.x, -result.fun)
                else:
                    return (None, None)
            except:
                return (None, None)
    
    def find_signal(self, reach=50, num=50):
        # Returns an array of frequencies around the peak frequency
        # with an array of their corresponding Ωh^2 values. The
        # 'reach' parameter determines the extreme frequencies.
        f_peak = self.find_peak()[0]
        f_min, f_max = (f_peak/reach, f_peak*reach)

        f_range = np.logspace(np.log10(f_min), np.log10(f_max), num=num)
        Omega_h2_range = self.Omega_turb_h2(f_range)

        return [f_range, Omega_h2_range]

    # # SOUND WAVE CONTRIBUTIONS
    # def f_sw_peak(self):
    #     return 1.9e-5 * (1/self.v_wall) * (self.beta_over_H) * (self.T_GW / 100 / GeV) * (self.g_eff_GW / 100)**(1/6)
    
    # def Omega_sw_h2(self, f):
    #     g_eff_PT = self.g_eff_GW
    #     v_w = self.v_wall
    #     beta_over_H = self.beta_over_H
    #     alpha = self.alpha
    #     kappa = self.kappa_v
    #     f_peak = self.f_sw_peak()

    #     return 2.62e-6 * v_w * (1/beta_over_H) * (kappa*alpha / (1+alpha))**2 * (g_eff_PT/100)**(-1/3) * \
    #     7**3.5 * (f / f_peak)**3 / (4 + 3*(f / f_peak)**2)**3.5
    
    # # BUBBLE COLLISION CONTRIBUTIONS
    # def f_col_peak(self):
    #     return 1.65e-5 * (0.62 / (1.8 - 0.1*self.v_wall + self.v_wall**2)) * self.beta_over_H * (self.T_GW / 100 / GeV) * (self.g_eff_GW / 100)**(1/6)

    # def Omega_col_h2(self, f):
    #     g_eff_PT = self.g_eff_GW
    #     v_w = self.v_wall
    #     beta_over_H = self.beta_over_H
    #     alpha = self.alpha
    #     kappa = self.kappa_col
    #     f_peak = self.f_col_peak()

    #     return 1.67e-5 * (0.11*v_w**3 / (0.42 + v_w**2)) * beta_over_H**(-2) * (kappa*alpha / (1 + alpha))**2 * (g_eff_PT / 100)**(-1/3) * \
    #     3.8*(f / f_peak)**2.8 / (1 + 0.28*(f / f_peak)**3.8)
    
    # # MAGNETOHYDRODYNAMIC TURBULENCE CONTRIBUTIONS
    # def f_turb_peak(self):
    #     return 2.7e-5 * (1 / self.v_wall) * self.beta_over_H * (self.T_GW / 100 / GeV) * (self.g_eff_GW / 100)**(1/6)
    
    # def Omega_turb_h2(self, f):
    #     g_eff_PT = self.g_eff_GW
    #     v_w = self.v_wall
    #     beta_over_H = self.beta_over_H
    #     alpha = self.alpha
    #     kappa = self.kappa_turb
    #     f_peak = self.f_turb_peak()

    #     h_star = 1.65e-5 * (self.T_GW / 100 / GeV) * (g_eff_PT / 100)**(1/6)

    #     return 3.35e-4 * v_w * beta_over_H**(-1) * (kappa*alpha / (1 + alpha))**(3/2) * (g_eff_PT / 100)**(-1/3) * \
    #     (f / f_peak)**3 / (1 + f / f_peak)**(11/3) / (1 + 8 * pi * f / h_star)
    
    # # TOTAL (assuming a linear combination)
    # def Omega_h2(self, f):
    #     return self.Omega_sw_h2(f) + self.Omega_col_h2(f) + self.Omega_turb_h2(f)
    
    # def find_peak(self, contribution='total'):
    #     # Returns a tuple (f_peak, Omega_h2_peak)
    #     if contribution == 'total':
    #         def Omega_h2(f): return self.Omega_h2(f)
    #     elif contribution == 'sw':
    #         def Omega_h2(f): return self.Omega_sw_h2(f)
    #     elif contribution == 'bc':
    #         def Omega_h2(f): return self.Omega_col_h2(f)
    #     elif contribution == 'mhd':
    #         def Omega_h2(f): return self.Omega_turb_h2(f)

    #     f_peaks = (self.f_col_peak(), self.f_sw_peak(), self.f_turb_peak())
    #     # negative frequencies?
    #     for index, freq in enumerate(f_peaks):
    #         if freq < 0:
    #             print(f"\nERROR: negative GW frequencies:\n \
    #                   f_peaks[{index}] = {freq}\n \
    #                   beta/H = {self.beta_over_H}")
    #             f_peaks[index] = np.real(freq)

    #     f_min, f_max = (min(f_peaks), max(f_peaks))
    #     bracket = [0.1*f_min, 10*f_max]
    #     result = minimize_scalar(lambda f: -Omega_h2(f), bracket=bracket)
    #     if result.success is True:
    #         return (result.x, -result.fun)
    #     else:
    #         return 'failed to find peak'
        