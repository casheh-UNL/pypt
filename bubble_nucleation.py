# Solves for bubble nucleation dynamic params based on bounce action and effective potential

# Here we take the characteristic temperature of the PT to be its
# nucleation temperature. Some authors (e.g. https://arxiv.org/pdf/2210.07075.pdf)
# use the percolation temperature when evaluating GW parameters.

from .constants import *
from .pt_math import *

from .ftpot import *

from cosmoTransitions.tunneling1D import SingleFieldInstanton
from .tunneling_potential import TunnelingPotential
from scipy.integrate import simpson

from .eff_DoF.eff_DoF import g_SM
# from scipy.misc import derivative # outdated
from numdifftools import Derivative as derivative

def g_rad(T):
    return g_SM(T) # + 2*1 + 1*3 + 1*1 + 1*1
def dg_rad(T):
    return derivative(g_rad, step=1e-2)(T)
def d2g_rad(T):
    return derivative(dg_rad, step=1e-2)(T)

class BubbleNucleationBL:
    def __init__(self, Veff: VeffBL, T_PT=-1, T_c=-1, METHOD='cosmoTransitions'):
        self.Veff = Veff
        self.T_c = self.Veff.find_critical_phi_and_temperature()[1] if T_c==-1 else T_c
    
        if METHOD in ('cosmoTransitions', 'TunnelingPotential'):
            self.METHOD = METHOD
        else:
            raise ValueError(f"Invalid METHOD: {METHOD}")
        self.TPT = self.find_nucleation_temperature() if T_PT==-1 else T_PT

    #############
    # COSMOLOGY #
    #############
    def rho_rad(self, T): return np.pi**2 / 30 * g_rad(T) * T**4
    def rho_vac(self, T):
        phi_min = self.Veff.find_minimum_phi(T) # true/broken vacuum
        deltaV = -self.Veff(phi_min, T) # Helmholtz free energy density
        dVdT = -self.Veff.dVeffBL_dT(phi_min, T) # entropy density s = -∂F/∂T...this should include phi(T) derivative???
        rho = deltaV - T*dVdT # E = F + TS
        return rho #!!! MUST BE MULTIPLIED BY THE FALSE VACUUM FRACTION...although it should be O(1) near the PT
    def rho(self, T):
        return self.rho_rad(T) + self.rho_vac(T)
    
    def hubble2(self, T):
        return 8*np.pi/3/M_PL**2 * self.rho(T) # using G = 1/M_Pl^2
    def ln_hubble2(self, T):
        return log(8) + log(np.pi) - log(3) - 2*log(M_PL) + log(self.rho(T))
    
    def dT_dt(self, T):
        phi_c = self.Veff.phi_min # should we use the false or true vacuum VEV?
        dVdT, d2VdT2 = self.Veff.dVeffBL_dT(phi_c, T), self.Veff.d2VeffBL_dT(phi_c, T)

        drhoRdT = (np.pi**2 / 30) * (dg_rad(T)*T + 4*g_rad(T)) * T**3
        d2rhoRdT2 = (np.pi**2 / 30) * (d2g_rad(T)*T**2 + 8*dg_rad(T)*T + 12*g_rad(T)) * T**2

        return -3 * np.sqrt(self.hubble2(T)) * (dVdT + drhoRdT/3) / (d2VdT2 + d2rhoRdT2/3)

    ####################
    # PHASE TRANSITION #
    ####################
    def bounce_action_cosmoTransitions(self, T):
        # returns S3 / T
        phi_absMin = self.Veff.find_minimum_phi(T)
        phi_metaMin = self.Veff.phi_min # cosmoTransitions is sensitive to this value

        SFI = SingleFieldInstanton(phi_absMin, phi_metaMin, V=lambda phi: self.Veff(phi, T), dV=lambda phi: self.Veff.dVeffBL(phi, T), d2V=lambda phi: self.Veff.d2VeffBL(phi, T))

        Tc = self.T_c

        # https://arxiv.org/pdf/2412.02645
        # p. 43 suggests phitol = 1e-10 = xtol for conformal models at low temperatures
        # ...but these values lead to errors about barrier heights: 'Barrier height is not positive, does not exist.', 'no barrier'
        # phitol, xtol = 1e-10, 1e-10
        phitol = 1e-5 * (T / Tc)
        # the default phitol=1e-4 is fine for high temperatures, but needs to be lowered for low temperatures
        profile = SFI.findProfile(phitol=phitol)#, xtol=xtol)

        # r, phi, dphi = profile.R, profile.Phi, profile.dPhi
        # integrand = 4*pi * r**2 * ( dphi**2 / 2 + self.Veff(phi, T) )
        # action = simpson(integrand, r)
        action = SFI.findAction(profile)

        return action / T
    
    def bounce_action_TunnelingPotential(self, T):
        TP = TunnelingPotential(self.Veff, self.Veff.dVeffBL, number_of_phis=200)
        phi0, action = TP.find_phi0_and_action(T, max_iterations=20, shift=0.5)
        return action / T
    
    def bounce_action(self, T):
        if self.METHOD == 'cosmoTransitions':
            try:
                return self.bounce_action_cosmoTransitions(T)
            except:
                # return a very large action so the rate effectively vanishes
                return np.inf
        elif self.METHOD == 'TunnelingPotential':
            return self.bounce_action_TunnelingPotential(T)

    def rate(self, T):
        S3_T = self.bounce_action(T)
        if not np.isfinite(S3_T) or S3_T <= 0:
            return 0.0
        return np.real(T**4 * power(abs(S3_T) / (2*pi), 3/2) * np.exp(-S3_T))
        # see Eq. (4.7) in https://arxiv.org/pdf/2210.07075.pdf
    def ln_rate(self, T):
        S3_T = self.bounce_action(T)
        if not np.isfinite(S3_T) or S3_T <= 0:
            return -np.inf

        ln_rate = 4*log(T) + (3/2)*(log(S3_T) - log(2*pi)) - S3_T
        return ln_rate

    
    def find_nucleation_temperature(self, TPT_guess=None):
        mu = self.Veff.mu
        Tc = self.T_c

        def nucleation_criterion(T):
            return self.ln_rate(T) - 2*self.ln_hubble2(T)
            # return (self.rate(T) / self.hubble2(T)**2) - 1.0
        # this is a very rough approximation...see Eq. (4.8) in https://arxiv.org/pdf/2210.07075.pdf
        # def integrand(T):
        #     return self.rate(T) / self.hubble2(T)**2 / T
        # def nucleation_criterion(Tn):
        #     return quad(integrand, Tn, Tc).y - 1.0
        # return root_scalar(nucleation_criterion, )

        def fit_func(x):
            # fits are mu- and method-dependent
            if self.METHOD == 'cosmoTransitions':
                if self.Veff.mu == 0.001:
                    stretch, squeeze, shift, max = (2.62766733,  3.48926794, -6.03223739, -0.45759643)
                elif self.Veff.mu == 0.01:
                    stretch, squeeze, shift, max = (3.78648288,  3.62542405, -6.57011118, -0.2395558)
                elif self.Veff.mu == 0.1:
                    stretch, squeeze, shift, max = (4.33810154,  3.5212693 , -6.41334754, -0.02752282)
                elif self.Veff.mu == 1.0:
                    stretch, squeeze, shift, max = (5.25782652,  4.98652449, -9.16113818, -0.39408839)
                elif self.Veff.mu == 10.0:
                    stretch, squeeze, shift, max = (5.66703201,  4.95681212, -9.10248055, -0.41316263)
                elif self.Veff.mu == 100.0:
                    stretch, squeeze, shift, max = (4.13643602,  2.0722626 , -3.30400284,  0.50583826)
                elif self.Veff.mu == 1000.0:
                    stretch, squeeze, shift, max = (4.13643602,  2.0722626 , -3.30400284,  0.50583826)
                elif self.Veff.mu == 10000.0:
                    stretch, squeeze, shift, max = (4.06798079,  3.33319106, -4.09095885, -0.24007116)
                else:
                    stretch, squeeze, shift, max = (4.33810154,  3.5212693 , -6.41334754, -0.02752282)
                return stretch * (np.arctan(squeeze*x - shift) - (np.pi/2 - max/stretch))
            
            elif self.METHOD == 'TunnelingPotential':
                if self.Veff.mu == 0.001:
                    stretch, squeeze, shift, max = (1.1508887 ,  2.92872716, -3.00585261,  0.20930855)
                elif self.Veff.mu == 0.01:
                    stretch, squeeze, shift, max = (1.1508887 ,  2.92872716, -3.00585261,  0.20930855)
                elif self.Veff.mu == 0.1:
                    stretch, squeeze, shift, max = (1.54523846,  6.23008058, -6.81439518, -0.23721182)
                elif self.Veff.mu == 1.0:
                    stretch, squeeze, shift, max = (2.35858842,  5.77477004, -6.33514384,  0.0573548)
                elif self.Veff.mu == 10.0:
                    stretch, squeeze, shift, max = (2.99973159,  5.18142685, -5.64599976,  0.32020058)
                elif self.Veff.mu == 100.0:
                    stretch, squeeze, shift, max = (3.54763294,  6.47368596, -6.89698061,  0.13788222)
                elif self.Veff.mu == 1000.0:
                    stretch, squeeze, shift, max = (4.98839816,  3.36837179, -3.72194462,  1.42705292)
                elif self.Veff.mu == 10000.0:
                    stretch, squeeze, shift, max = (4.61038255,  4.45755422, -4.84854708,  0.07259731)
                return stretch * (np.arctan(squeeze*x - shift) - (np.pi/2 - max/stretch))
        if TPT_guess == None:
            TPT_guess = mu * np.exp(fit_func(np.log(Tc / mu)))
        
        # attempt Brent's method with a loose bracket
        try:
            T_low = 1e-5 * Tc
            T_high = 0.99999 * Tc # nucleation_criterion(Tc) = NaN (due to a constant (zero) tunneling potential leaving S3 undefined)
            result = root_scalar(nucleation_criterion, method='brentq', bracket=[T_low, T_high])
            if result.converged:
                return result.root
        except (ValueError, RuntimeError):
            pass # failed bracket or non-convergence
        # fall back to secant method
        result = root_scalar(nucleation_criterion, x0=TPT_guess, x1=0.9*TPT_guess)
        if result.converged and not np.isclose(result.root, TPT_guess, rtol=1e-4):
            return result.root # avoid trivial convergence to initial guess
        else:
            return None

    def alpha(self, TPT=None):
        # latent heat
        if TPT is None:
            TPT = self.TPT

        prefactor = 30 / pi**2 / g_rad(TPT) / TPT**4
        phi_min = self.Veff.find_minimum_phi(TPT)

        deltaV = -self.Veff(phi_min, TPT) # false - true
        # dVdT = (self.Veff(phi_min, 1.001*TPT) - self.Veff(phi_min, 0.999*TPT)) / (1.001 * TPT - 0.999 * TPT)
        dVdT = -self.Veff.dVeffBL_dT(phi_min, TPT)
        # in the limit of large supercooling (TPT << T_c), entropy contributions can be neglected https://arxiv.org/pdf/2210.07075.pdf
        
        return prefactor * (deltaV - TPT/4 * dVdT)

    def betaByHstar(self):
        # assumes log(Γ) ~ β(t-t_PT)
        TPT = self.TPT
        S3_T = self.bounce_action(TPT)
        dS3_T_dT = (self.bounce_action(1.001*TPT) - self.bounce_action(0.999 * TPT)) / (1.001 * TPT - 0.999 * TPT)
        dT_dt = self.dT_dt(TPT)

        beta = -dS3_T_dT*dT_dt + 4*dT_dt/TPT + (3/2)*dT_dt*dS3_T_dT/S3_T
        H = np.sqrt(self.hubble2(TPT))
        
        return beta / H

    def vw(self):
        alpha = self.alpha()
        return (1/sqrt(3) + sqrt(alpha**2 + 2*alpha/3)) / (1+alpha)
    

    ##########################
    # MEAN BUBBLE SEPARATION #
    ##########################
    # When log(Γ) doesn't grow linearly with time, the above betaByHstar is not a good a good
    # approximation for the characteristic time scale of the PT.
    # Instead, the mean bubble separation R_* ~ n^{-1/3} is used. An analogous time scale can
    # then be obtained via β/H := (8π)^{1/3} vw / (R_* H) ~ (8π n)^{1/3} vw / H .
    # NOTE: n BELOW GIVES THE NUMBER DENSITY OF TRUE/BROKEN BUBBLES, SO IT CANNOT BE USED IN
    # PBH FORMATION CALCULATIONS.

    def T_interval(self, T1, T2, num, dist='log'):
        '''
        Temperature interval over which numerical integrations are done.
        '''
        if dist=='log':
            return np.logspace(np.log10(T1), np.log10(T2), num=num, endpoint=False, base=10.)
        elif dist=='lin':
            return np.linspace(T1, T2, num=num, endpoint=False)
    def numerical_integral(self, T1, T2, integrand, num, dist='log', TEST=False):
        '''
        Helper function for performing the integrations over temperature from T1 to T2.
        Integration bounds have been swapped and a minus sign has been included in the integrand. This is to keep from evaluating
        functions at the critical temperature T1 = T_crit.
        '''
        Ts = self.T_interval(T2, T1, num, dist)
        integrand_vals = np.array([-integrand(T) for T in Ts])
        integral = simpson(y=integrand_vals, x=Ts)
        if TEST:
            return (integral, integrand_vals, Ts)
        else:
            return integral
        
    def a_ratio(self, Tp, T, num=20):
        '''
        a(T') / a(T) in a flat FLRW universe
        num=20 yields about 1.3% error
        '''
        phi_c = self.Veff.phi_min
        def drho_dT(T):
            # assumes dV/dT = ∂V/∂T
            drhoR_dT = (np.pi**2 / 30) * (dg_rad(T)*T + 4*g_rad(T)) * T**3
            d2V_dT = self.Veff.d2VeffBL_dT(phi_c, T)
            return drhoR_dT - T*d2V_dT
        def rho_plus_P(T):
            dV_dT = self.Veff.dVeffBL_dT(phi_c, T)
            return (4/3)*self.rho_rad(T) - T*dV_dT
        
        def integrand(T):
            return (-1/3) * (drho_dT(T) / rho_plus_P(T))
        integral = self.numerical_integral(T, Tp, integrand, num)
        return np.exp(integral)
    
    def R_integrand(self, T, Tpp, vw):
        return (1/self.dT_dt(Tpp)) * self.a_ratio(T, Tpp) * vw
    def R(self, Tp, T, vw, R0=0, num=100):
        '''
        Radius at T of a bubble nucleated at Tp with initial radius R0.
        num=100 yields about 1.0% error
        '''
        def integrand(Tpp):
            return self.R_integrand(T, Tpp, vw)
        integral = self.numerical_integral(Tp, T, integrand, num)
        return self.a_ratio(T, Tp) * R0 + integral
    def V(self, Tp, T, vw, R0=0):
        return (4*np.pi/3) * self.R(Tp, T, vw, R0)**3

    def Pf_integrand(self, T, Tp, vw, TEST=False):
        dt_dT, Gamma, aRatio_cubed, volume = (1/self.dT_dt(Tp)) * self.rate(Tp) * self.a_ratio(Tp, T)**3 * self.V(Tp, T, vw)
        integrand = dt_dT * Gamma * aRatio_cubed * volume
        if TEST:
            return (dt_dT, Gamma, aRatio_cubed, volume, integrand)
        else:
            return integrand
    def Pf(self, T, vw, num=100, TEST=False):
        '''
        Fraction of 3-space within the false vacuum.
        '''
        def integrand(Tp):
            return self.Pf_integrand(T, Tp, vw)
        integral = self.numerical_integral(self.T_c, T, integrand, num=num, TEST=TEST)
        if TEST:
            integral_val, integrand_vals, Ts = integral
            return (np.exp(-integral_val), integral_val, integrand_vals, Ts)
        else:
            return np.exp(-integral)
    
    def n_integrand(self, T, Tp, vw, num=100, TEST=False):
        dt_dT, Gamma, aRatio_cubed, Pf = (1/self.dT_dt(Tp)), self.rate(Tp), self.a_ratio(Tp, T)**3, self.Pf(Tp, vw, num=num)
        integrand = dt_dT * Gamma * aRatio_cubed * Pf
        if TEST:
            H = np.sqrt(self.hubble2(Tp))
            return (dt_dT, Gamma, aRatio_cubed, Pf, integrand, H)
        else:
            return integrand
    def n(self, T, vw, num=50, TEST=False):
        '''
        Number density of true-vacuum bubbles.
        '''
        def integrand(Tp):
            return self.n_integrand(T, Tp, vw)
        integral = self.numerical_integral(self.T_c, T, integrand, num=num, TEST=TEST)
        if TEST:
            return (integral[0], integral[1], integral[2])
        else:
            return integral
    


    # APPROXIMATION: radiation domination with constant vw and g_*
    def H_rad(self, T):
        return np.sqrt(self.hubble2(T))[0]
        # return np.sqrt(8*np.pi/3/M_PL**2 * self.rho_rad(T))
    
    def R_rad(self, Tp, T, vw, R0=0):
        # radius at T of a bubble nucleated at Tp with initial radius R0
        def R_integrand(Tpp):
            return vw / T / self.H_rad(Tpp)
        Ts = np.logspace(np.log10(T), np.log10(Tp), num=50, endpoint=False, base=10.)
        integrand_vals = [R_integrand(T) for T in Ts]
        return simpson(y=integrand_vals, x=Ts)
        # return quad(R_integrand, T, Tp)[0]
        # c = np.sqrt(45 * M_PL**2 / 4 / np.pi**3 / g_rad(T)) * vw
        # return (c / T) * (1/T - 1/Tp) + (Tp/T)*R0
    def V_rad(self, Tp, T, vw, R0=0):
        return (4*np.pi/3) * self.R_rad(Tp, T, vw, R0)**3

    def T_interval_rad(self, T, num, dist='log'):
        '''
        Temperature interval over which numerical integrations are done.
        '''
        if dist=='log':
            return np.logspace(np.log10(T), np.log10(self.T_c), num=num, endpoint=False, base=10.)
        elif dist=='lin':
            return np.linspace(T, self.T_c, num=num, endpoint=False)
        
    def Pf_integrand_rad(self, Tp, T, vw):
        return self.rate(Tp) * T**3 * self.V_rad(Tp, T, vw) / self.H_rad(Tp) / Tp**4
    def Pf_rad(self, T, vw, num=50, TEST=False):
        # fraction of 3-space within the false vacuum
        Ts = self.T_interval_rad(T, num, dist='log')
        integrand_vals = np.array([self.Pf_integrand_rad(Tp, T, vw) for Tp in Ts]).flatten()
        integral = simpson(y=integrand_vals, x=Ts)
        if TEST:
            # put into your code near where you compute integrand_vals for Pf_rad(TEST=True)
            Ts = self.T_interval_rad(T, num, dist='log')
            out = []
            for Tp in Ts:
                S3 = self.bounce_action(Tp)
                # protect against solver failures
                if not np.isfinite(S3) or S3 <= 0:
                    lnGamma = -np.inf
                    Gamma = 0.0
                else:
                    # compute ln(Gamma) safely
                    lnGamma = 4*np.log(Tp) + 1.5*(np.log(abs(S3)) - np.log(2*np.pi)) - S3
                    # if you want Gamma itself (may overflow)
                    Gamma = np.exp(lnGamma)  

                Htp = self.H_rad(Tp)  # make sure H_rad returns scalar float (not array)
                R = self.R_rad(Tp, T, vw)   # your physical radius at time T
                V = (4*np.pi/3) * R**3

                prefactor = np.nan
                ln_prefactor = np.nan
                if Htp > 0 and np.isfinite(Htp) and Gamma>0:
                    # integrand = Gamma * T**3 / (Htp * Tp**4) * V
                    ln_prefactor = lnGamma + 3*np.log(T) - (np.log(Htp) + 4*np.log(Tp)) + np.log(V)
                    try:
                        integrand = np.exp(ln_prefactor)
                    except OverflowError:
                        integrand = np.inf
                else:
                    integrand = 0.0
                    ln_prefactor = -np.inf

                out.append((Tp, S3, lnGamma, Gamma, Htp, R, V, ln_prefactor, integrand))

            # Pretty-print a few rows
            for row in out[:12]:
                Tp, S3, lnGamma, Gamma, Htp, R, V, ln_prefactor, integrand = row
                print(f"Tp={Tp:.6e}  S3={S3:.4e}  lnGamma={lnGamma:.4e}  H={Htp:.4e}  R={R:.4e}  ln(integrand)={ln_prefactor:.4e}  integrand={integrand:.4e}")
            return (np.exp(-integral), integral, Ts, integrand_vals)
        else:
            return np.exp(-integral)
    
    def n_integrand_rad(self, T, Tp, vw, num, TEST=False):
        if TEST:
            H = self.H_rad(Tp)
            dt_dT, Gamma, Pf = (-1/H/Tp), self.rate(Tp), self.Pf_rad(Tp, vw, num=num)
            integrand = Gamma * Tp**3 * Pf / H / Tp**4
            return (dt_dT, Gamma, Pf, integrand, H)
        else:
            return self.rate(Tp) * T**3 * self.Pf_rad(T, vw, num=num) / self.H_rad(Tp) / Tp**4
    def n_rad(self, T, vw, num=50, num_Pf=50, TEST=False):
        # number density of true-vacuum bubbles
        Ts = self.T_interval_rad(T, num, dist='log')            
        integrand_vals = np.array([self.n_integrand_rad(T, Tp, vw, num_Pf) for Tp in Ts])
        integral = simpson(y=integrand_vals, x=Ts)
        if TEST:
            return (integral, Ts, integrand_vals)
        else:
            return integral