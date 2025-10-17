# Class for storing a finite temp effective potential

from .constants import *
from .cosmology_functions import *
from .pt_math import *

##### B-L #####
from .pt_math import *

# class for specifying particle information relevant to thermal potentials
class Field(object):
    def __init__(self, dof=0, mass_squared=None, dmass_squared=None, dmass_squared_dT=None, Debye_mass_squared=None, dDebye_mass_squared=None, dDebye_mass_squared_dT=None, type='', name=''):
        self.dof = dof # particle degrees of freedom
        self.mass_squared = mass_squared # particle mass squared
        self.dmass_squared = dmass_squared # phi derivative of mass squared
        self.dmass_squared_dT = dmass_squared_dT # T derivative of mass squared
        self.Debye_mass_squared = Debye_mass_squared # Debye mass squared for bosons
        self.dDebye_mass_squared = dDebye_mass_squared # phi derivative of Debye mass squared
        self.dDebye_mass_squared_dT = dDebye_mass_squared_dT # T derivative of Debye mass squared
        self.type = type # 'boson' or 'fermion'
        self.name = name
    
    def __str__(self):
        return f'field: {self.name} // type: {self.type} // DoF = {self.dof} // mass squared: {self.mass_squared} // d(mass squared)/dphi: {self.dmass_squared}'

    def __repr__(self):
        return self.__str__()


class VeffBL():
    # https://arxiv.org/pdf/2502.19478
    def __init__(self, g=0.7, y=0.3, mu=0.1, phi_min=1e-6, phi_max=1.4):
        self.g = g
        self.y = y
        self.lam = self.lam_value()
        self.mu = mu
        self.phi_min = phi_min * self.mu
        self.phi_max = phi_max * self.mu # should be greater than mu to allow t = 0

        # self.t_values = self.RGE_solutions()[0]
        # self.sol1 = self.RGE_solutions()[1]
        # self.sol2 = self.RGE_solutions()[2]
        # self.phi_min = np.exp(min(self.t_values)) # RGE solver may not make it down to self.phi_min
        # self.Landau_pole = True if self.phi_max != phi_min*self.mu else False

        self.field_list = [self.RHN1(),
                           self.Zprime(),
                           self.Phi(),
                           self.G()
                           ]
        self.some_data = []
        #self.phi_c, self.Tc = self.find_critical_phi_and_temperature()
            

    def alpha_BL(self, t):
        t = np.asarray(t)
        return pi * self.alpha_BL0 / (pi - 6 * t * self.alpha_BL0)
    
    
    # renormalization group equations (RGEs) for ONE RHN...need to check dalpha_Y
    def RGE(self, t, alphas):
        alpha_lambda, alpha_Y = alphas
        dalpha_lambda = (10 * alpha_lambda**2 + alpha_lambda * \
                            (alpha_Y - 24 * self.alpha_BL(t)) + \
                            48 * self.alpha_BL(t)**2 - \
                            0.5 * alpha_Y**2) / 2 / pi
        dalpha_Y = (alpha_Y * (alpha_Y - 18 * self.alpha_BL(t))) / 2 / pi
        return [dalpha_lambda, dalpha_Y]
    
    # RGEs "initial" conditions
    # def alpha_Y0(self): 
        # return self.k**2 * self.alpha_BL0
        # assumes the Majorana Yukawa and B-L couplings are proportional:
        # Y = k g_{B-L}

    def lam_value(self):
        # eq. (2.7)
        root1 = (24*pi**2 + sqrt(-3630*self.g**4 + 576*pi**4 + 1210*self.y**4)) / 110
        root2 = (24*pi**2 - sqrt(-3630*self.g**4 + 576*pi**4 + 1210*self.y**4)) / 110
        # both roots have positive values for some g,y but root1 has more positive values
        return root2
    
    # RGEs solutions using solve_ivp...would solve_bvp work better?
    def RGE_solutions(self):

        # t_eval may not be necessary
        t_eval1 = np.linspace(self.t(self.mu, T=0), self.t(self.phi_min, T=0), 200)
        t_eval2 = np.linspace(self.t(self.mu, T=0), self.t(self.phi_max, T=10*self.mu), 20)
        
        # halt integration if |αλ| > 0.25
        def alpha_lambda_event(t, y):
            return abs(y[0]) - 0.25
        alpha_lambda_event.terminal = True # stop integration when this occurs
        alpha_lambda_event.direction = 0   # detect zero-crossing in either direction

        # sol1 gives solutions for t in [log(phi_min), 0]
        # sol2 gives solutions for t in [0, log(phi_max)]
        sol1 = solve_ivp(self.RGE, [self.t(self.mu, T=0), self.t(self.phi_min, T=0)], [self.alpha_lambda0(), self.alpha_Y0], t_eval=t_eval1, dense_output=True, events=[alpha_lambda_event,])
        sol2 = solve_ivp(self.RGE, [self.t(self.mu, T=0), self.t(self.phi_max, T=10*self.mu)], [self.alpha_lambda0(), self.alpha_Y0], t_eval=t_eval2, dense_output=True)
        t_values = np.concatenate((sol1.t[::-1], sol2.t[1:]))
        return [t_values, sol1, sol2]

    def alpha_lambda(self, t):
        alpha_lambda_values = np.concatenate((self.sol1.y[0,::-1], self.sol2.y[0,1:]))
        return interp1d(self.t_values, alpha_lambda_values, fill_value="extrapolate")(t)
        # !! be careful using "extrapolate"...https://stackoverflow.com/questions/45429831/valueerror-a-value-in-x-new-is-above-the-interpolation-range-what-other-re
    def alpha_Y(self, t):
        # alpha_Y_values = np.concatenate((self.sol1.y[1,::-1], self.sol2.y[1,1:]))
        # alphaY = interp1d(self.t_values, alpha_Y_values, fill_value="extrapolate")(t)

        t = np.array(t)
        alphaY = (30*pi * self.alpha_BL0 * (pi - 6*t*self.alpha_BL0)**(3/2) * self.alpha_Y0) \
                    / (pi**(5/2) * (30*self.alpha_BL0 - self.alpha_Y0) + (pi - 6*t*self.alpha_BL0)**(5/2) * self.alpha_Y0)
        return alphaY

    def t(self, phi, T=0):
        # technically need to use |phi|
        return log(phi / self.mu)
        # return log(np.sqrt(phi**2 + T**2) / self.mu)
        # return log( np.maximum(phi, T) / self.mu)
    # derivative with respect to φ
    def dt(self, phi, T):
        phi, T = np.array(phi), np.array(T)
        i = True
        # i = (phi >= T)
        return (1 / phi) * i
        # return phi / (phi**2 + T**2)
    # derivative with respect to T
    def dtdT(self, phi, T):
        phi, T = np.array(phi), np.array(T)
        i = True
        # i = (phi >= T)
        return 0.

    # derivatives with respect to t
    def dalpha_lambda(self, t):
        return derivative(self.alpha_lambda, t, 1e-6)
    def dalpha_Y(self, t):
        num = -3*pi**(5/2) * np.sqrt(pi - 6*t*self.alpha_BL0) * (30*self.alpha_BL0 - self.alpha_Y0) + 2*(pi - 6*t*self.alpha_BL0)**3 * self.alpha_Y0
        den = pi**(5/2) * (30*self.alpha_BL0 - self.alpha_Y0) + (pi - 6*t*self.alpha_BL0)**(5/2) * self.alpha_Y0
        return 90*pi*self.alpha_BL0**2 * self.alpha_Y0 * num / den**2
        # return (self.alpha_Y(t) * (self.alpha_Y(t) - 18 * self.alpha_BL(t))) / 2 / pi
    def dalpha_BL(self, t):
        return 6*pi*self.alpha_BL0**2 / (pi - 6*t*self.alpha_BL0)**2
    
    
    # particles...DoF are doubled to account for antiparticles
    def RHN1(self):
        return Field(dof=2*2, mass_squared=self.m2_RHN1, dmass_squared=self.dm2_RHN1, dmass_squared_dT=self.dm2_dT_RHN1, type='fermion', name=r'$\nu_R$')
    def Zprime(self):
        return Field(3, self.m2_Zprime, self.dm2_Zprime, self.dm2_dT_Zprime, self.Pi_Zprime, self.dPi_Zprime, self.dPi_dT_Zprime, 'boson', r'$Z^\prime$')
    def Phi(self):
        return Field(1, self.m2_Phi, self.dm2_Phi, self.dm2_dT_Phi, self.Pi_Phi, self.dPi_Phi, self.dPi_dT_Phi, 'boson', r'$\phi$')
    def G(self):
        return Field(1, self.m2_G, self.dm2_G, self.dm2_dT_G, self.Pi_G, self.dPi_G, self.dPi_dT_G, 'boson', r'$G$')

    # squared particle masses  
    def m2_RHN1(self, phi):  # right-handed neutrino
        phi = np.array(phi)
        return self.y**2 * phi**2 / 2
    
    def m2_Zprime(self, phi): # B-L gauge boson
        phi = np.array(phi)
        return self.g**2 * phi**2
    
    def m2_Phi(self, phi): # B-L symmetry-breaking scalar
        phi = np.array(phi)
        return 3 * self.lam * phi**2
    
    def m2_G(self, phi): # Goldstone boson
        phi = np.array(phi)
        return self.lam * phi**2
    
    # phi-derivatives of squared masses
    def dm2_RHN1(self, phi):
        phi= np.array(phi)
        dm2_dphi = self.y**2 * phi
        return dm2_dphi
    def dm2_Zprime(self, phi):
        phi= np.array(phi)
        dm2_dphi = 2* self.g**2 * phi
        return dm2_dphi
    def dm2_Phi(self, phi):
        phi= np.array(phi)
        dm2_dphi = 6 * self.lam * phi
        return dm2_dphi
    def dm2_G(self, phi):
        phi= np.array(phi)
        dm2_dphi = 2 * self.lam * phi
        return dm2_dphi
    # T-derivatives of squared masses
    def dm2_dT_RHN1(self, phi, T):
        phi, T = np.array(phi), np.array(T)
        return 0
    def dm2_dT_Zprime(self, phi, T):
        phi, T = np.array(phi), np.array(T)
        return 0
    def dm2_dT_Phi(self, phi, T):
        phi, T = np.array(phi), np.array(T)
        return 0
    def dm2_dT_G(self, phi, T):
        phi, T = np.array(phi), np.array(T)
        return 0
    
    # Debye masses of bosons
    def Pi_Zprime(self, T):
        T = np.array(T)
        return (5/12) * self.g**2 * T**2
    def Pi_Phi(self, T):
        T = np.array(T)
        return (self.lam/3 + self.y**2/12 + self.g**2/4) * T**2
    def Pi_G(self, T):
        return self.Pi_Phi(T)
    
    # phi-derivatives of squared Debye masses
    def dPi_Zprime(self, T):
        T = np.array(T)
        return 0
    def dPi_Phi(self, T):
        T = np.array(T)
        return 0
    def dPi_G(self, T):
        return self.dPi_Phi(T)
    # T-derivatives of squared Debye masses
    def dPi_dT_Zprime(self, T):
        T = np.array(T)
        return (5/6) * self.g**2 * T
    def dPi_dT_Phi(self, T):
        T = np.array(T)
        return 2 * (self.lam/3 + self.y**2/12 + self.g**2/4) * T
    def dPi_dT_G(self, phi, T):
        return self.dPi_dT_Phi(T)


    #################################################################################
    # potentials...only the real parts
    def Veff0_no_shift(self, phi): # zero-temperature tree-level + Coleman-Weinberg potential
        # eq. (2.8)
        phi = np.array(phi)
        v = self.mu
        def boson_sum(phi):
            return sum(field.dof/64/pi**2 * 1 * field.mass_squared(v)**2/v**4 * phi**4 * (np.log(phi**2/v**2) - 1/2) \
                    for field in self.field_list if field.type=='boson')
        def fermion_sum(phi):
            return sum(field.dof/64/pi**2 * (-1) * field.mass_squared(v)**2/v**4 * phi**4 * (np.log(phi**2/v**2) - 1/2) \
                    for field in self.field_list if field.type=='fermion')
      
        return np.real( boson_sum(phi) + fermion_sum(phi) )
    
    def Veff0(self, phi):
        return self.Veff0_no_shift(phi) - self.Veff0_no_shift(self.phi_min)

    def _V_T(self, phi, T):
        """Calculates the one-loop thermal correction V_T."""
        # This term is zero if T=0, which the T**4 prefactor handles automatically.
        # We use np.where to avoid division by zero when an element of T is 0.
        T_squared = T**2
        T_squared_safe = np.where(T_squared > 0, T_squared, 1.0)

        # Use list comprehensions summed by NumPy. This is efficient and clean.
        boson_sum = np.sum([
            field.dof * J_B(field.mass_squared(phi) / T_squared_safe)
            for field in self.field_list if field.type == 'boson'
        ], axis=0)

        fermion_sum = np.sum([
            field.dof * J_F(field.mass_squared(phi) / T_squared_safe)
            for field in self.field_list if field.type == 'fermion'
        ], axis=0)

        prefactor = T**4 / (2 * pi**2)
        return prefactor * (boson_sum - fermion_sum)

    def _V_daisy(self, phi, T):
        """Calculates the daisy resummation correction."""
        # This entire correction should vanish at T=0. This happens if
        # Debye_mass_squared(T) is proportional to T.

        # Warning: This logic relies on hardcoded list indices (e.g., [1], [2:]),
        # making it fragile. Consider identifying fields by name or another attribute.
        Zprime = self.field_list[1]
        other_bosons = self.field_list[2:]

        def power_diff(m_sq_debye, m_sq):
            # Helper to safely compute (m_debye^2)^1.5 - (m^2)^1.5
            # Using np.power handles potential complex numbers from negative m_sq.
            return np.power(m_sq_debye, 1.5) - np.power(m_sq, 1.5)

        # Sum over the 'other bosons'
        daisy_sum_term = np.sum([
            field.dof * power_diff(
                field.mass_squared(phi) + field.Debye_mass_squared(T),
                field.mass_squared(phi)
            )
            for field in other_bosons
        ], axis=0)

        # Special term for only including the longitudinal component of Z'
        zprime_term = 1 * power_diff(
            Zprime.mass_squared(phi) + Zprime.Debye_mass_squared(T),
            Zprime.mass_squared(phi)
            )

        return (-T / (12 * pi)) * (daisy_sum_term + zprime_term)

    def Veff_no_shift(self, phi, T):
        """
        Calculates the finite-temperature effective potential using vectorization.
        """
        # Ensure inputs are NumPy arrays and apply Z_2 symmetry.
        phi_arr = np.asanyarray(np.abs(phi))
        T_arr = np.asanyarray(T)

        # The T=0 potential should already be vectorized.
        V0 = self.Veff0_no_shift(phi_arr)

        # For T=0, the thermal corrections are zero. We can short-circuit
        # for the simple scalar case, but the helpers also handle T=0 correctly.
        if np.all(T_arr == 0):
            return V0

        # Calculate thermal and daisy corrections. These helper functions
        # are designed to handle arrays and will return 0 for elements where T=0.
        V_T_corr = self._V_T(phi_arr, T_arr)
        V_daisy_corr = self._V_daisy(phi_arr, T_arr)

        V_total = V0 + V_T_corr + V_daisy_corr

        # Taking np.real() is good practice, as thermal functions can yield
        # small imaginary parts if mass-squared arguments become negative.
        return np.real(V_total)

    def __call__(self, phi, T):
        return self.Veff_no_shift(phi, T) - self.Veff_no_shift(self.phi_min, T)
    


    ###################
    # PHI DERIVATIVES #
    ###################
    def dVeff0(self, phi):
        g, y, lam, v = self.g, self.y, self.lam, self.mu
        return (3*g**4 - y**4 + 10*lam**2) * phi**3 * np.log(phi**2 / v**2) / (16 * pi**2)
    
    def dV_T(self, phi, T):
        # This term is zero if T=0, which the T**4 prefactor handles automatically.
        # We use np.where to avoid division by zero when an element of T is 0.
        T_squared = T**2
        T_squared_safe = np.where(T_squared > 0, T_squared, 1.0)

        # Use list comprehensions summed by NumPy. This is efficient and clean.
        boson_sum = np.sum([
            field.dof * dJ_B(field.mass_squared(phi) / T_squared_safe) * field.dmass_squared(phi)
            for field in self.field_list if field.type == 'boson'
        ], axis=0)

        fermion_sum = np.sum([
            field.dof * dJ_F(field.mass_squared(phi) / T_squared_safe) * field.dmass_squared(phi)
            for field in self.field_list if field.type == 'fermion'
        ], axis=0)

        prefactor = T**4 / (2 * pi**2)
        return prefactor * (boson_sum - fermion_sum)

    def dV_daisy(self, phi, T):
        # This entire correction should vanish at T=0. This happens if
        # Debye_mass_squared(T) is proportional to T.

        # Warning: This logic relies on hardcoded list indices (e.g., [1], [2:]),
        # making it fragile. Consider identifying fields by name or another attribute.
        Zprime = self.field_list[1]
        other_bosons = self.field_list[2:]

        def power_diff(m_sq_debye, m_sq):
            # Helper to safely compute (m_debye^2)^0.5 - (m^2)^0.5
            # Using np.power handles potential complex numbers from negative m_sq.
            return np.power(m_sq_debye, 0.5) - np.power(m_sq, 0.5)

        # Sum over the 'other bosons'
        daisy_sum_term = np.sum([
            field.dof * power_diff(
                field.mass_squared(phi) + field.Debye_mass_squared(T),
                field.mass_squared(phi)
            ) * field.dmass_squared(phi)
            for field in other_bosons
        ], axis=0)

        # Special term for only including the longitudinal component of Z'
        zprime_term = 1 * power_diff(
            Zprime.mass_squared(phi) + Zprime.Debye_mass_squared(T),
            Zprime.mass_squared(phi)
            ) * Zprime.dmass_squared(phi)

        return (-T / (8 * pi)) * (daisy_sum_term + zprime_term)
    
    def dVeffBL(self, phi, T):
        """
        Calculates the finite-temperature effective potential using vectorization.
        """
        # Ensure inputs are NumPy arrays and apply Z_2 symmetry.
        phi_arr = np.asanyarray(np.abs(phi))
        T_arr = np.asanyarray(T)

        # The T=0 potential should already be vectorized.
        dV0 = self.dVeff0(phi_arr)

        # For T=0, the thermal corrections are zero. We can short-circuit
        # for the simple scalar case, but the helpers also handle T=0 correctly.
        if np.all(T_arr == 0):
            return dV0

        # Calculate thermal and daisy corrections. These helper functions
        # are designed to handle arrays and will return 0 for elements where T=0.
        dV_T_corr = self.dV_T(phi_arr, T_arr)
        dV_daisy_corr = self.dV_daisy(phi_arr, T_arr)

        dV_total = dV0 + dV_T_corr + dV_daisy_corr

        # Taking np.real() is good practice, as thermal functions can yield
        # small imaginary parts if mass-squared arguments become negative.
        return np.real(dV_total)
    
    
    # second derivative of Veff with respect to phi
    def d2VeffBL(self, phi, T=0):
        phi = np.atleast_1d(np.abs(phi)) # potential has Z_2 symmetry
        derivs = np.array([
            derivative(lambda Phi: self.dVeffBL(Phi, T), step=1e-6*self.mu)(phi_val)
            for phi_val in phi
            ])
        return derivs[0] if derivs.size == 1 else derivs
    
    # derivative of Veff with respect to T
    def dVeffBL_dT(self, phi, T=0):
        phi = np.atleast_1d(np.abs(phi))
        T = np.atleast_1d(T)
        derivs = np.array([
            derivative(lambda temp: self.__call__(phi, temp), step=1e-6*self.mu)(Temp)
            for Temp in T
            ])

        return np.real( derivs[0] ) if derivs.size == 1 else np.real(derivs)
    
    # second derivative of Veff with respect to T
    def d2VeffBL_dT(self, phi, T=0):
        phi = np.atleast_1d(np.abs(phi))
        T = np.atleast_1d(T)
        derivs = np.array([
            derivative(lambda temp: self.dVeffBL_dT(phi, temp), step=1e-6*self.mu)(Temp)
            for Temp in T
        ])

        return derivs[0] if derivs.size == 1 else derivs




    
    def find_minimum_phi(self, T):
        guess = self.mu
        min_phi_array = root(lambda phi: self.dVeffBL(phi, T), guess).x
        return min_phi_array[0]
    
    
    def find_critical_phi_and_temperature(self, init=None):
        # true_mu = self.mu
        # self.mu = 1 # here, we exploit conformal symmetry rescale the potential to mu=1, find the critical values, then scale back to self.mu
        mu, phi_max = self.mu, self.phi_max
        def T_crit_guess():
            # these guesses comes from numerical studies of successful scans
            alpha_BL0 = self.g**2 / 4 / pi
            if alpha_BL0 < 1e-2:
                return 10**(0.25) * (alpha_BL0)**(0.5) * mu
            else:
                return 10**(0.55) * (alpha_BL0)**(0.645475532172732) * mu
        if init is None:
            init = (mu, T_crit_guess())
        def roots(func, init):
            solutions = root(func, init) # tol=self.mu * 1e-9
            tolerance = abs(solutions.fun) <= 1e-5 * np.array([self(phi_max, T_crit_guess()) / mu**4, self.dVeffBL(phi_max, T_crit_guess()) / mu**3])
            if (solutions.success is True) and (tolerance.all() == True):
                return solutions.x
            else:
                return None # 'critical values not found...try changing the initial guesses init=(phi0, T0)'
            
        def V_dV(x):
            phi, T = x
            V = self(phi, T) / mu**4
            dV = self.dVeffBL(phi, T) / mu**3
            # scaling is to tame values for extreme μ
            return [V, dV]
  
        result = roots(V_dV, init) # returns array([phi_crit, T_crit])
        # self.mu = true_mu
        if (result is not None) and (result[0] > 0.1*mu) and (result[1] > 0):
            return result
        else:
            return None
        

    def plot_potential(self, T, phi_range=None, phi_steps=100):
        T = np.atleast_1d(T)
        import matplotlib.pyplot as plt

        if phi_range is None:
            phi_range = (self.phi_min, self.phi_max)

        phis = np.linspace(phi_range[0], phi_range[1], phi_steps)

        for temp in T:
            plt.plot(phis, self(phis, temp), label=rf'$T = ${temp:.3e}')

        plt.title(rf'$\alpha_{{B-L}}(0) = ${self.alpha_BL0:.3e}, $\alpha_{{Y}}(0) = ${self.alpha_Y0:.3e}')
        plt.xlabel(r'$\phi$')
        plt.ylabel(r'$V_{\rm eff}(\phi,T)$')

        plt.legend()
        plt.grid()
        plt.show()
    
    def find_surface_tension(self):
        # bubble surface tension, which is calculated at the critical temperature
        phi_c, T_c = self.find_critical_phi_and_temperature()

        # generate points for integration
        phi_values = np.linspace(self.phi_min, phi_c, num=1001)  # Simpson's rule requires an odd number of points

        func_values = np.real(np.sqrt(2*self.__call__(phi_values, T_c) + 0j))
        return simpson(func_values, phi_values)

    def find_bubble_critical_radius(self):
        # critical radius R_c = 2σ / ΔV where σ is the bubble surface tension
        phi_c, T_c = self.find_critical_phi_and_temperature()

        sigma = self.find_surface_tension()
        dV = self.__call__(phi_c, T_c) # usually taken at T = T_c, but Lewicki et al. do not include thermal effects in ΔV
        return abs(2*sigma / dV)

    def find_critical_radius_estimate(self):
        # critical radius R_c = 2σ / ΔV where σ is the bubble surface tension
        phi_c, T_c = self.find_critical_phi_and_temperature()
        phi_min = self.find_minimum_phi(T_c)
        mPhi = np.sqrt(self.m2_Phi(phi_c, T_c))
        mG = np.sqrt(self.m2_G(phi_c, T_c))
        mZp = np.sqrt(self.m2_Zprime(phi_c, T_c))

        A = (1 / 12 / np.pi / phi_c**3) * (mPhi**3 + mG**3 + 3*mZp**3)
        sigma = (2**(3/2) / 3**4) * (A**3 / self.alpha_lambda(self.t(phi_c, T_c))**(5/2)) * T_c**3
        dV = self.__call__(phi_min, T_c)

        return 2*sigma / dV