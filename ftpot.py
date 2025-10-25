from .pt_math import *

# class for specifying particle information relevant to thermal potentials
class Field(object):
    def __init__(self, dof=0, mass_squared=None, dmass_squared=None, dmass_squared_dT=None,
                 Debye_mass_squared=None, dDebye_mass_squared=None, dDebye_mass_squared_dT=None,
                 type='', name='', has_longitudinal_mode=False):
        self.dof = dof # particle degrees of freedom
        self.mass_squared = mass_squared # particle mass squared
        self.dmass_squared = dmass_squared # phi derivative of mass squared
        self.dmass_squared_dT = dmass_squared_dT # T derivative of mass squared
        self.Debye_mass_squared = Debye_mass_squared # Debye mass squared for bosons
        self.dDebye_mass_squared = dDebye_mass_squared # phi derivative of Debye mass squared
        self.dDebye_mass_squared_dT = dDebye_mass_squared_dT # T derivative of Debye mass squared
        self.type = type # 'boson' or 'fermion'
        self.name = name
        self.has_longitudinal_mode = has_longitudinal_mode # for Arnold-Espinosa thermal resummation
    
    def __str__(self):
        return f'field: {self.name} // type: {self.type} // DoF = {self.dof} // mass squared: {self.mass_squared} // d(mass squared)/dphi: {self.dmass_squared}'

    def __repr__(self):
        return self.__str__()


class VeffBL(object):
    def __init__(self, alpha_BL0=0.01, alpha_Y0=0.1, mu=1.0, phi_min=1e-6, phi_max=1.4):
        self.alpha_BL0 = alpha_BL0
        self.alpha_Y0 = alpha_Y0
        # self.k = k
        self.mu = mu
        self.phi_min = phi_min * self.mu
        self.phi_max = phi_max * self.mu # should be greater than mu to allow t = 0

        self.t_values = self.RGE_solutions()[0]
        self.sol1 = self.RGE_solutions()[1]
        self.sol2 = self.RGE_solutions()[2]
        self.phi_min = np.exp(min(self.t_values)) # RGE solver may not make it down to self.phi_min
        self.Landau_pole = True if self.phi_max != phi_min*self.mu else False

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

    def alpha_lambda0(self): # Eq. (16) of https://arxiv.org/pdf/0902.4050.pdf
        return (-4 * pi + sqrt(16 * pi**2 + 5 * (-96 * self.alpha_BL0**2 + self.alpha_Y0**2))) / 10
    
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
        # return derivative(self.alpha_lambda, t, 1e-6)
        return Derivative(self.alpha_lambda, step=1e-6)(t)
    def dalpha_Y(self, t):
        num = -3*pi**(5/2) * np.sqrt(pi - 6*t*self.alpha_BL0) * (30*self.alpha_BL0 - self.alpha_Y0) + 2*(pi - 6*t*self.alpha_BL0)**3 * self.alpha_Y0
        den = pi**(5/2) * (30*self.alpha_BL0 - self.alpha_Y0) + (pi - 6*t*self.alpha_BL0)**(5/2) * self.alpha_Y0
        return 90*pi*self.alpha_BL0**2 * self.alpha_Y0 * num / den**2
        # return (self.alpha_Y(t) * (self.alpha_Y(t) - 18 * self.alpha_BL(t))) / 2 / pi
    def dalpha_BL(self, t):
        return 6*pi*self.alpha_BL0**2 / (pi - 6*t*self.alpha_BL0)**2
    
    
    # particles...double DoF if antiparticles give distinct thermal contributions
    def RHN1(self): # Majorana fermions are their own antiparticles
        return Field(dof=2, mass_squared=self.m2_RHN1, dmass_squared=self.dm2_RHN1, dmass_squared_dT=self.dm2_dT_RHN1, type='fermion', name=r'$\nu_R$')
    def Zprime(self): # gauge bosons are real-valued so cannot have antiparticles
        # While massless, vector bosons should have only 2 DoF...do thermal (Debye) masses inform these DoF?
        return Field(3, self.m2_Zprime, self.dm2_Zprime, self.dm2_dT_Zprime, self.Pi_Zprime, self.dPi_Zprime, self.dPi_dT_Zprime, 'boson', r'$Z^\prime$',
                     has_longitudinal_mode=True)
    def Phi(self): # this is the real component after symmetry breaking
        return Field(1, self.m2_Phi, self.dm2_Phi, self.dm2_dT_Phi, self.Pi_Phi, self.dPi_Phi, self.dPi_dT_Phi, 'boson', r'$\phi$')
    def G(self): # again, real field after symmetry breaking
        return Field(1, self.m2_G, self.dm2_G, self.dm2_dT_G, self.Pi_G, self.dPi_G, self.dPi_dT_G, 'boson', r'$G$')

    # squared particle masses  
    def m2_RHN1(self, phi, T):  # right-handed neutrino
        phi, T = np.array(phi), np.array(T)
        return (4 * pi * self.alpha_Y( self.t(phi,T) )) * phi**2 / 2
    
    def m2_Zprime(self, phi, T): # B-L gauge boson
        phi, T = np.array(phi), np.array(T)
        return 4 * (4 * pi * self.alpha_BL( self.t(phi,T) )) * phi**2
    
    def m2_Phi(self, phi, T): # B-L symmetry-breaking scalar
        phi, T = np.array(phi), np.array(T)
        return 3 * (4 * pi * self.alpha_lambda( self.t(phi,T) )) * phi**2
    
    def m2_G(self, phi, T): # Goldstone boson
        phi, T = np.array(phi), np.array(T)
        return (4 * pi * self.alpha_lambda( self.t(phi,T) )) * phi**2
    
    # phi-derivatives of squared masses
    def dm2_RHN1(self, phi, T):
        phi, T = np.array(phi), np.array(T)
        dm2_dphi = 2 * pi * self.dalpha_Y(self.t(phi, T)) * self.dt(phi, T) * phi**2 + \
            4 * pi * self.alpha_Y(self.t(phi, T)) * phi
        return dm2_dphi
    def dm2_Zprime(self, phi, T):
        phi, T = np.array(phi), np.array(T)
        dm2_dphi = 16 * pi * self.dalpha_BL(self.t(phi, T)) * self.dt(phi, T) * phi**2 + \
            32 * pi * self.alpha_BL(self.t(phi, T)) * phi
        return dm2_dphi
    def dm2_Phi(self, phi, T):
        phi, T = np.array(phi), np.array(T)
        dm2_dphi = 12 * pi * self.dalpha_lambda(self.t(phi, T)) * self.dt(phi, T) * phi**2 + \
            24 * pi * self.alpha_lambda(self.t(phi, T)) * phi
        return dm2_dphi
    def dm2_G(self, phi, T):
        phi, T = np.array(phi), np.array(T)
        dm2_dphi = 4 * pi * self.dalpha_lambda(self.t(phi, T)) * self.dt(phi, T) * phi**2 + \
            8 * pi * self.alpha_lambda(self.t(phi, T)) * phi
        return dm2_dphi
    # T-derivatives of squared masses
    def dm2_dT_RHN1(self, phi, T):
        phi, T = np.array(phi), np.array(T)
        dm2_dT = 2 * pi * self.dalpha_Y(self.t(phi, T)) * self.dtdT(phi, T) * phi**2
        return dm2_dT
    def dm2_dT_Zprime(self, phi, T):
        phi, T = np.array(phi), np.array(T)
        dm2_dT = 16 * pi * self.dalpha_BL(self.t(phi, T)) * self.dtdT(phi, T) * phi**2
        return dm2_dT
    def dm2_dT_Phi(self, phi, T):
        phi, T = np.array(phi), np.array(T)
        dm2_dT = 12 * pi * self.dalpha_lambda(self.t(phi, T)) * self.dtdT(phi, T) * phi**2
        return dm2_dT
    def dm2_dT_G(self, phi, T):
        phi, T = np.array(phi), np.array(T)
        dm2_dT = 4 * pi * self.dalpha_lambda(self.t(phi, T)) * self.dtdT(phi, T) * phi**2
        return dm2_dT
    
    # Debye masses of bosons
    def Pi_Zprime(self, phi, T):
        phi, T = np.array(phi), np.array(T)
        return 16 * pi * self.alpha_BL(self.t(phi, T)) * T**2
    def Pi_Phi(self, phi, T):
        phi, T = np.array(phi), np.array(T)
        return (4*self.alpha_BL(self.t(phi,T)) + 4*self.alpha_lambda(self.t(phi,T))/3 * self.alpha_Y(self.t(phi,T))/6) * pi * T**2
    def Pi_G(self, phi, T):
        return self.Pi_Phi(phi, T)
    
    # phi-derivatives of squared Debye masses
    def dPi_Zprime(self, phi, T):
        phi, T = np.array(phi), np.array(T)
        return 16*pi * T**2 * self.dalpha_BL(self.t(phi,T)) * self.dt(phi,T)
    def dPi_Phi(self, phi, T):
        phi, T = np.array(phi), np.array(T)
        return ( 4*self.dalpha_BL(self.t(phi,T)) + 4*self.dalpha_lambda(self.t(phi,T))/3 + self.dalpha_Y(self.t(phi,T))/6 ) * self.dt(phi,T) * pi * T**2
    def dPi_G(self, phi, T):
        return self.dPi_Phi(phi,T)
    # T-derivatives of squared Debye masses
    def dPi_dT_Zprime(self, phi, T):
        phi, T = np.array(phi), np.array(T)
        return 16*pi * (T**2 * self.dalpha_BL(self.t(phi,T)) * self.dtdT(phi,T) + 2*self.alpha_BL(self.t(phi,T))*T)
    def dPi_dT_Phi(self, phi, T):
        phi, T = np.array(phi), np.array(T)
        return ( 4*self.dalpha_BL(self.t(phi,T)) + 4*self.dalpha_lambda(self.t(phi,T))/3 + self.dalpha_Y(self.t(phi,T))/6 ) * self.dtdT(phi,T) * pi * T**2 \
                + (8*self.alpha_BL(self.t(phi,T)) + 8*self.alpha_lambda(self.t(phi,T))/3 + self.alpha_Y(self.t(phi,T))/3) * pi * T
    def dPi_dT_G(self, phi, T):
        return self.dPi_dT_Phi(phi,T)


    #################################################################################
    # potentials...only the real parts
    def Veff0_no_shift(self, phi): # zero-temperature RG-improved potential
        phi, T = np.array(phi), np.array(0)

        # def negative_gamma(t): # anomalous dimension for one RHN
        #     return -(self.alpha_Y(t) - 24 * self.alpha_BL(t)) / 8 / pi
        # def G(t):
        #     try:
        #         return exp( quad(negative_gamma, 0.0, t)[0] )
        #     except:
        #         return np.array( [exp (quad(negative_gamma, 0., t_)[0]) for t_ in t] ).reshape(np.shape(t))
        
        def V0(phi):
            # return pi * self.alpha_lambda(self.t(phi, T)) * G(self.t(phi, T))**4 * phi**4
            # more analytic expression:
            return np.sqrt(pi)*phi**4 * self.alpha_lambda(self.t(phi, T)) * \
                (self.alpha_Y0*(pi - 6*self.alpha_BL0*self.t(phi, T))**(5/2) + pi**(5/2)*(30*self.alpha_BL0 - self.alpha_Y0)) \
                / (30*self.alpha_BL0 * (pi - 6*self.alpha_BL0*self.t(phi, T))**2)
      
        return np.real( V0(phi) )
    
    def Veff0(self, phi):
        return self.Veff0_no_shift(phi) - self.Veff0_no_shift(self.phi_min)


    def V_daisy_AE(self, field, phi, T):
        """
        Arnold–Espinosa daisy correction term for a single bosonic field.
        Only the longitudinal polarization receives thermal mass correction.

        ΔV_daisy = - (T / 12π) * ñ * [ (m² + Π)^(3/2) - m³ ]
        """
        m2 = field.mass_squared(phi, T)
        d2 = field.Debye_mass_squared(phi, T)

        # Only apply to modes that have a longitudinal component
        if not getattr(field, "has_longitudinal_mode", False):
            return np.zeros_like(phi, dtype=float)

        n_tilde = 1.0  # longitudinal DoF
        m = np.sqrt(np.maximum(m2, 0.0))
        mT = np.sqrt(np.maximum(m2 + d2, 0.0))

        return - (T / (12 * np.pi)) * n_tilde * (mT**3 - m**3)

    def V_1B(self, phi, T, thermal_resum='Parwani'):
        '''
        One-loop, nonzero temperature contributions to the effective potential from real boson fields, vectorized.
        Parameters
        ----------
        phi : array_like
            Field value(s).
        T : float
            Temperature.
        thermal_resum : str, optional
            Daisy resummation scheme. Options:
            - 'Parwani' : Replace m^2 → m^2 + Π(T) everywhere (all DoF resummed)
            - 'Arnold-Espinosa' : Add separate daisy correction term for
                                longitudinal bosonic modes only.
        '''
        # This logic assumes T != 0, which is handled by the caller.
        T2 = T**2
        total_sum = np.zeros_like(phi, dtype=float)

        for field in self.field_list:
            if field.type == 'boson':
                # ASSUMPTION: These functions are NumPy-aware
                m2 = field.mass_squared(phi, T)
                d2 = field.Debye_mass_squared(phi, T)
                # --- Parwani scheme ---
                if thermal_resum.lower() == 'parwani':
                    # All bosonic degrees of freedom get resummed mass
                    arg = (m2 + d2) / T2
                    total_sum += (T**4 / 2 / pi**2) * field.dof * np.real(J_B(arg))

                # --- Arnold–Espinosa scheme ---
                elif thermal_resum.lower() == 'arnold-espinosa':
                    # Thermal function evaluated with unresummed mass
                    arg = m2 / T2
                    total_sum += (T**4 / 2 / pi**2) * field.dof * np.real(J_B(arg))

                    # Add the separate Daisy correction (longitudinal modes only)
                    total_sum += self.V_daisy_AE(field, phi, T)

        return total_sum

    def V_1F(self, phi, T):
        '''
        One-loop, nonzero temperature contributions to the effective potential from fermion fields, vectorized.
        '''
        T2 = T**2
        total_sum = np.zeros_like(phi, dtype=float)

        for field in self.field_list:
            if field.type == 'fermion':
                # ASSUMPTION: These functions are NumPy-aware
                m2 = field.mass_squared(phi, T)
                arg = m2/T2
                
                # ASSUMPTION: J_F is NumPy-aware
                total_sum += field.dof * np.real(J_F(arg))
                
        return - (T**4 / 2 / pi**2) * total_sum
        # Some B-L literature forgets this negative sign
        # e.g. https://arxiv.org/abs/1811.11169 eq. (8)
        # e.g. https://arxiv.org/abs/2007.15586 eq. (2.2)
    
    def dV1B_dphi(self, phi, T):
        """Helper to calculate the derivative of the bosonic thermal sum w/rt phi, vectorized."""
        T2 = T**2
        total_dsum = np.zeros_like(phi, dtype=float)

        for field in self.field_list:
            if field.type == 'boson':
                # These functions are assumed to be NumPy-aware
                m2 = field.mass_squared(phi, T)
                d2 = field.Debye_mass_squared(phi, T)
                dm2 = field.dmass_squared(phi, T)       # Derivative of m^2 w/rt phi
                dd2 = field.dDebye_mass_squared(phi, T) # Derivative of Debye mass^2 w/rt phi
                
                arg = m2/T2 + d2/T2
                darg_dphi = (dm2 + dd2) / T2  # Derivative of the argument w/rt phi
                
                # Apply the chain rule: d/dphi [ J_B(arg) ] = dJ_B/darg * darg/dphi
                # ASSUMPTION: dJ_B is NumPy-aware and returns dJ_B/darg
                term = field.dof * np.real(dJ_B(arg)) * darg_dphi
                total_dsum += term
                
        return (T**4 / 2 / pi**2) * total_dsum

    def dV1F_dphi(self, phi, T):
        """Helper to calculate the derivative of the fermionic thermal sum w/rt phi, vectorized."""
        T2 = T**2
        total_dsum = np.zeros_like(phi, dtype=float)

        for field in self.field_list:
            if field.type == 'fermion':
                # These functions are assumed to be NumPy-aware
                m2 = field.mass_squared(phi, T)
                dm2 = field.dmass_squared(phi, T)   # Derivative of m^2 w/rt phi
                
                arg = m2/T2
                darg_dphi = dm2 / T2    # Derivative of the argument w/rt phi
                
                # Apply the chain rule: d/dphi [ J_F(arg) ] = dJ_F/darg * darg/dphi
                # ASSUMPTION: dJ_F is NumPy-aware and returns dJ_F/darg
                term = field.dof * np.real(dJ_F(arg)) * darg_dphi
                
                total_dsum += term
                
        return - (T**4 / 2 / pi**2) * total_dsum
    
    def Veff_no_shift(self, phi, T):
        phi = np.abs(np.asarray(phi)) # Z_2 symmetry
        T = np.asarray(T)

        # Broadcast inputs to a common shape
        try:
            phi_b, T_b = np.broadcast_arrays(phi, T)
        except ValueError as e:
            raise ValueError(f"phi and T shapes cannot be broadcast: {phi.shape} and {T.shape}") from e

        # ASSUMPTION: self.Veff0_no_shift is NumPy-aware
        # If not, you must vectorize it:
        # v_Veff0 = np.vectorize(self.Veff0_no_shift)
        # V0 = v_Veff0(phi_b)
        V0 = self.Veff0_no_shift(phi_b)
        
        # Initialize the full T-dependent potential
        V_T = np.array(V0, dtype=float)
        
        # Create mask for T > 0
        mask = (T_b > 0)
        
        # Only compute sums where T > 0
        if np.any(mask):
            # Find values at the masked locations
            phi_masked = phi_b[mask]
            T_masked = T_b[mask]
            
            # Calculate sums only for these masked values
            b_sum = self.V_1B(phi_masked, T_masked)
            f_sum = self.V_1F(phi_masked, T_masked)
            
            # Calculate thermal potential only for these values
            V_T_masked = b_sum + f_sum
            
            # Add these values back into the full-size array
            V_T[mask] += V_T_masked
                
        # The potential values are usually real since the thermal integrals in pt_math.py have their
        # imaginary components discarded, but for whatever reason, scipy's solvers work better with the
        # np.real() output
        return np.real(V_T)
        
    def __call__(self, phi, T):
        return self.Veff_no_shift(phi, T) - self.Veff_no_shift(self.phi_min, T)


    # derivative of Veff with respect to phi
    def dVeffBL(self, phi, T=0):
        phi = np.abs(phi) # potential has Z_2 symmetry
        phi, T = np.array(phi), np.array(T)

        # def negative_gamma(t): # anomalous dimension for one RHN
        #     return -(self.alpha_Y(t) - 24 * self.alpha_BL(t)) / 8 / pi
        # def G(t):
        #     try:
        #         return exp( quad(negative_gamma, 0.0, t)[0] )
        #     except:
        #         return np.array( [exp(quad(negative_gamma, 0., t_)[0]) for t_ in t] ).reshape(np.shape(t))

        # dV_dphi = 4 * pi * self.alpha_lambda(self.t(phi, T)) * G(self.t(phi, T))**4 * phi**3 + \
        #           pi * (self.dalpha_lambda(self.t(phi, T))) * G(self.t(phi, T))**4 * self.dt(phi, T) * phi**4 + \
        #           pi * self.alpha_lambda(self.t(phi, T)) * 4*G(self.t(phi, T))**4 * negative_gamma(self.t(phi, T)) * self.dt(phi, T) * phi**4 + \
        #           (T**2 / 2 / pi**2) * (dboson_sum(phi, T) + dfermion_sum(phi, T))

        # more analytic expression?
        alpha_BL0, alpha_Y0 = self.alpha_BL0, self.alpha_Y0
        term = pi - 6*alpha_BL0*self.t(phi, T)
        dV0_dphi = np.sqrt(pi)*phi**3 / (30*alpha_BL0* term**3) * \
            (term * phi * self.dt(phi, T) * (alpha_Y0* term**(5/2) + pi**(5/2)*(30*alpha_BL0 - alpha_Y0)) * self.dalpha_lambda(self.t(phi, T)) \
             - 15*alpha_BL0*alpha_Y0 * term**(5/2) * self.alpha_lambda(self.t(phi, T)) * phi * self.dt(phi, T) \
                + 4*term*self.alpha_lambda(self.t(phi, T)) * (alpha_Y0* term**(5/2) + pi**(5/2)*(30*alpha_BL0 - alpha_Y0)) \
                    + 12*alpha_BL0*self.alpha_lambda(self.t(phi, T)) * phi * self.dt(phi, T) * (alpha_Y0*term**(5/2) + pi**(5/2)*(30*alpha_BL0 - alpha_Y0)))

        dV_dphi = dV0_dphi + self.dV1B_dphi(phi, T) + self.dV1F_dphi(phi, T)

        # handle T = 0 cases
        result = np.where(T==0, dV0_dphi, dV_dphi)
        
        return np.real( result )
    
    # second derivative of Veff with respect to phi
    def d2VeffBL(self, phi, T=0):
        phi = np.atleast_1d(np.abs(phi)) # potential has Z_2 symmetry
        derivs = np.array([
            # derivative(lambda Phi: self.dVeffBL(Phi, T), phi_val, 1e-6*self.mu)
            Derivative(lambda Phi: self.dVeffBL(Phi, T), step=1e-6*self.mu)(phi_val)
            for phi_val in phi
            ])
        return derivs[0] if derivs.size == 1 else derivs
    
        
    # derivative of Veff with respect to T
    def dVeffBL_dT(self, phi, T=0):
        phi = np.abs(phi) # potential has Z_2 symmetry
        phi, T = np.array(phi), np.atleast_1d(T)

        def gamma(t): # anomalous dimension for one RHN
            return (self.alpha_Y(t) - 24 * self.alpha_BL(t)) / 8 / pi
        def G4(t):
            alpha_BL0, alpha_Y0 = self.alpha_BL0, self.alpha_Y0
            term = pi - 6*alpha_BL0*t

            num = pi**(5/2) * (30*alpha_BL0 - alpha_Y0) + term**(5/2) * alpha_Y0
            den = 30*np.sqrt(pi) * alpha_BL0 * term**2
            return num / den
        

        def boson_sum(phi, T):
            return sum(field.dof * J_B(field.mass_squared(phi,T)/T**2 + field.Debye_mass_squared(phi,T)/T**2) \
                    for field in self.field_list if field.type=='boson')
        def fermion_sum(phi, T):
            return sum(field.dof * J_F(field.mass_squared(phi,T)/T**2) \
                    for field in self.field_list if field.type=='fermion')
        def dboson_sum(phi, T):
            return sum(field.dof * dJ_B(field.mass_squared(phi,T)/T**2 + field.Debye_mass_squared(phi,T)/T**2) * (T**2*(field.dmass_squared_dT(phi, T) + field.dDebye_mass_squared_dT(phi, T)) - 2*T*(field.mass_squared(phi,T) + field.Debye_mass_squared(phi,T)))\
                    for field in self.field_list if field.type=='boson')
        def dfermion_sum(phi, T):
            return sum(field.dof * dJ_F(field.mass_squared(phi,T)/T**2) * (T**2*field.dmass_squared_dT(phi, T) - 2*T*field.mass_squared(phi,T)/T**2)\
                    for field in self.field_list if field.type=='fermion')

        def dV0_dT(phi,T):
            return pi*(self.dalpha_lambda(self.t(phi,T)) - 4*self.alpha_lambda(self.t(phi,T))*gamma(self.t(phi,T))) * G4(self.t(phi,T))*self.dtdT(phi,T)*phi**4
        def dV_dT(phi,T):
            return dV0_dT(phi,T) + (2*T**3 / pi**2)*(boson_sum(phi,T) + fermion_sum(phi,T)) + (1/2/pi**2)*(dboson_sum(phi, T) + dfermion_sum(phi, T))
        # dV0 = dV0_dT(phi,T) - dV0_dT(self.phi_min, T)
        # dV = dV_dT(phi,T) - dV_dT(self.phi_min, T)

        # handle T = 0 cases
        # result = np.where(T==0, dV0, dV)
        # I don't know why, but the above analytic expression isn't matching with numerical calculations of derivatives.
        # So we'll use the numerical derivatives for now:
        derivs = np.array([
            # derivative(lambda temp: self.__call__(phi, temp), Temp, 1e-6*self.mu)
            Derivative(lambda temp: self.__call__(phi, temp), step=1e-6*self.mu)(Temp)
            for Temp in T
            ])

        return np.real( derivs[0] ) if derivs.size == 1 else np.real(derivs)
    
    # second derivative of Veff with respect to T
    def d2VeffBL_dT(self, phi, T=0):
        phi = np.abs(phi) # potential has Z_2 symmetry
        phi, T = np.array(phi), np.atleast_1d(T)
        derivs = np.array([
            # derivative(lambda temp: self.dVeffBL_dT(phi, temp), temp, dx=1e-6 * self.mu)
            Derivative(lambda temp: self.dVeffBL_dT(phi, temp), step=1e-6*self.mu)(Temp)
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
            if self.alpha_BL0 < 1e-2:
                return 10**(0.25) * (self.alpha_BL0)**(0.5) * mu
            else:
                return 10**(0.55) * (self.alpha_BL0)**(0.645475532172732) * mu
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