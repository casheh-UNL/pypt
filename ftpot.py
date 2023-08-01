# Class for storing a finite temp effective potential

from .constants import *
from .pt_math import *


PT_CONST_AB = 16 * pi**2 * exp(1.5 - 2*GAMMA_EULER)
PT_CONST_AF = pi**2 * exp(1.5 - 2*GAMMA_EULER)
PT_CONST_EXPAB = exp(log(PT_CONST_AB) - 1.5)
PT_CONST_EXPAF = exp(log(PT_CONST_AB) - 1.5)


def jf(m, T):
    pass




class VFT:
    def __init__(self):
        self.T0 = None
        self.Tc = None

    def a2(self, T):
        return 1.0

    def a3(self, T):
        return 1.0

    def a4(self, T):
        return 1.0

    def phi_plus(self, T):
        return np.real(-3*self.a3(T) + sqrt(9*self.a3(T)**2 - 32*self.a4(T)*self.a2(T)))/(8*self.a4(T))

    def get_T0(self, T_range=[1.0, 100.0]):
        res = fsolve(self.a2, T_range)
        return abs(res[0])
    
    def get_Tc(self):
        res = fsolve(self.Veff0Min, [self.T0, 2*self.T0])
        return abs(res[1])
    
    def Veff0Min(self, T):
        return self.a2(T) * self.phi_plus(T)**2 + self.a3(T) * self.phi_plus(T)**3 + self.a4(T) * self.phi_plus(T)**4

    def __call__(self, phi, T):
        pass





class VEffSM(VFT):
    def __init__(self):
        self.Tc = 150.0
        
        self.D0 = (2*M_W**2 + M_Z**2 + 2*M_T**2) / 8 / VEV_H**2
        self.D1 = (2 * M_W**3 + M_Z**3) / (4*pi*VEV_H**3)
        self.D2 = 3 * (2*M_W**4 + M_Z**4 - 4*M_T**4) / (64*pi**2 * VEV_H**4)
        self.T0 = sqrt((M_H**2 - 8*self.D2*VEV_H**2)/(4*self.D0))


    def lamT(self, T):
        return 0.5*power(M_H/VEV_H, 2) - 3*(2*M_W**4 * log(M_W**2 / PT_CONST_EXPAB / T**2) \
            + M_Z**4 * log(M_Z**2 / PT_CONST_EXPAB / T**2) \
            - 4*M_T**4 * log(M_T**2 / PT_CONST_EXPAF / T**2))/(16*pi**2 * VEV_H**4)

    def __call__(self, phi, T):
        return np.real(self.D0*(self.Tc**2 - T**2)*phi**2 - self.D1*T*phi**3 + self.lamT(T)*phi**4)




class VEffMarfatia(VFT):
    def __init__(self, gchi=1.0, mchi=1.0, mu=100.0, lam=0.1, c=0.1, Lambda=1000.0):
        super().__init__()
        self.gchi = gchi
        self.mchi = mchi
        self.mu = mu
        self.lam = lam
        self.c = c
        self.Lambda = Lambda
    
    def set_params(self, gchi=1.0, mchi=1.0, mu=100.0, lam=0.1, c=0.1, Lambda=1000.0):
        self.gchi = gchi
        self.mchi = mchi
        self.mu = mu
        self.lam = lam
        self.c = c
        self.Lambda = Lambda
    
    def a2(self, T):
        beta = 1/T
        return 1/(1536*pi**2*beta**4*self.mu**8)*(-36*self.gchi**4*beta**4*self.mu**8+64*pi**2*beta**4*self.lam*self.mu**8-9*beta**4*self.lam**2*self.mu**8-3*self.c**4*pi*(beta**2*self.mu**2)**(3/2)-12*pi*self.lam**2*self.mu**4*(beta**2*self.mu**2)**(3/2)-24*self.gchi**4*beta**4*self.mu**8*log((self.mchi**2*beta**2)/PT_CONST_AF)+24*self.gchi**4*beta**4*self.mu**8*log(self.mchi**2/self.Lambda**2)-6*beta**4*self.lam**2*self.mu**8*log((beta**2*self.mu**2)/PT_CONST_AB)+6*beta**4*self.lam**2*self.mu**8*log(self.mu**2/self.Lambda**2))

    def a3(self, T):
        beta = 1/T
        return -(1/(192*pi**2*beta**4*self.mu**6))*(18*self.gchi**3*self.mchi*beta**4*self.mu**6-32*self.c*pi**2*beta**4*self.mu**6-self.c**3*pi*(beta**2*self.mu**2)**(3/2)+12*self.gchi**3*self.mchi*beta**4*self.mu**6*log((self.mchi**2*beta**2)/PT_CONST_AF)-12*self.gchi**3*self.mchi*beta**4*self.mu**6*log(self.mchi**2/self.Lambda**2))

    def a4(self, T):
        beta = 1/T
        return (1/(384*pi**2*beta**4*self.mu**4))*(-8*self.gchi**2*pi**2*beta**2*self.mu**4-9*self.c**2*beta**4*self.mu**4-54*self.gchi**2*self.mchi**2*beta**4*self.mu**4+8*pi**2*beta**2*self.lam*self.mu**4+192*pi**2*beta**4*self.mu**6-9*beta**4*self.lam*self.mu**6-12*self.c**2*pi*(beta**2*self.mu**2)**(3/2)-24*pi*self.lam*self.mu**2*(beta**2*self.mu**2)**(3/2)-36*self.gchi**2*self.mchi**2*beta**4*self.mu**4*((self.mchi**2*beta**2)/PT_CONST_AF)+36*self.gchi**2*self.mchi**2*beta**4*self.mu**4*(self.mchi**2/self.Lambda**2)-6*self.c**2*beta**4*self.mu**4*((beta**2*self.mu**2)/PT_CONST_AB)-6*beta**4*self.lam*self.mu**6*((beta**2*self.mu**2)/PT_CONST_AB)+6*self.c**2*beta**4*self.mu**4*(self.mu**2/self.Lambda**2)+6*beta**4*self.lam*self.mu**6*(self.mu**2/self.Lambda**2))
        
    def __call__(self, phi, T):
        return np.real(self.a2(1/T)*phi**2 + self.a3(1/T)*phi**3 + self.a4(1/T)*phi**4)




class VEffMarfatia2(VFT):
    def __init__(self, a=0.1, lam=0.061, c=0.249, d=0.596, b=75.0**4):
        super().__init__()
        self.a = a
        self.b = b
        self.lam = lam
        self.c = c
        self.d = d
        self.T0 = self.get_T0()
        self.Tc = self.get_Tc()
    
    def phi_plus0(self, T0):
        return (3*self.c + sqrt(9*self.c**2 + 8*self.lam*self.d*T0**2))/(2*self.lam)
    
    def get_T0(self):
        def root_func(T0):
            return self.b - 0.5 * self.phi_plus0(T0)**2 * (self.d*T0**2 + 0.5*self.c*self.phi_plus0(T0))
        
        res = fsolve(root_func, [1.0])
        return res[0]
    
    def a2(self, T):
        return self.d * (T**2 - self.T0**2)
    
    def a3(self, T):
        return -(self.a*T + self.c)

    def a4(self, T):
        return 0.25*self.lam
    
    def set_params(self, a=0.1, lam=0.061, c=0.249, d=0.596):
        self.a = a
        self.lam = lam
        self.c = c
        self.d = d
        self.T0 = self.get_T0()
        self.Tc = self.get_Tc()

    def __call__(self, phi, T):
        return np.real(self.d * (T**2 - self.T0**2)*phi**2 - (self.a*T + self.c)*phi**3 + 0.25*self.lam*phi**4)

##### B-L #####

# class for specifying particle information relevant to thermal potentials
class Field(object):
    def __init__(self, dof=0, mass_squared=None, dmass_squared=None, type='', name=''):
        self.dof = dof # particle degrees of freedom
        self.mass_squared = mass_squared # particle mass squared
        self.dmass_squared = dmass_squared # phi derivative of mass squared
        self.type = type # 'boson' or 'fermion'
        self.name = name
    
    def __str__(self):
        return f'field: {self.name} // type: {self.type} // DoF = {self.dof} // mass squared: {self.mass_squared}'

    def __repr__(self):
        return self.__str__()


class VeffBL(VFT):
    def __init__(self, alpha_BL0=0.01, k=0.1, phi_min=1e-6, phi_max=1.4, mu=1):
        self.alpha_BL0 = alpha_BL0
        self.k = k
        self.phi_min = phi_min
        self.phi_max = phi_max # should be greater than mu to allow t = 0
        self.mu = mu

        self.t_values = self.RGE_solutions()[0]
        self.sol1 = self.RGE_solutions()[1]
        self.sol2 = self.RGE_solutions()[2]

        self.field_list = [self.RHN1(),
                           self.Zprime(),
                           self.Phi(),
                           self.G()]
            

    def alpha_BL(self, t):
        return pi * self.alpha_BL0 / (pi - 6 * t * self.alpha_BL0)
    
    
    # renormalization group equations (RGEs) for ONE RHN...need to check dalpha_Y
    def RGE(self, t, alphas): # if solve_ivp doesn't like 'self', insert this in RGE_solutions
        alpha_lambda, alpha_Y = alphas
        dalpha_lambda = (10 * alpha_lambda**2 + alpha_lambda * \
                            (alpha_Y - 24 * self.alpha_BL(t)) + \
                            48 * self.alpha_BL(t)**2 - \
                            0.5 * alpha_Y**2) / 2 / pi
        dalpha_Y = (alpha_Y * (alpha_Y - 18 * self.alpha_BL(t))) / 2 / pi
        return [dalpha_lambda, dalpha_Y]
    
        # RGEs "initial" conditions
    def alpha_Y0(self):
        return self.k**2 * self.alpha_BL0
        # we assume the Majorana Yukawa and B-L couplings are proportional:
        # Y = k g_{B-L}

    def alpha_lambda0(self): # Eq. (16) of https://arxiv.org/pdf/0902.4050.pdf
        return (-4 * pi + sqrt(16 * pi**2 + 5 * (-96 * self.alpha_BL0**2 + self.alpha_Y0()**2))) / 10
    
        # RGEs solutions using solve_ivp...would solve_bvp work better?
    def RGE_solutions(self):

        # t_eval may not be necessary
        t_eval1 = np.linspace(log(1), log(self.phi_min), 100)
        t_eval2 = np.linspace(log(1), log(self.phi_max), 10)
        
        # sol1 gives solutions for t in [log(phi_min), 0]
        # sol2 gives solutions for t in [0, log(phi_max)]
        sol1 = solve_ivp(self.RGE, [log(1), log(self.phi_min)], [self.alpha_lambda0(), self.alpha_Y0()], t_eval=t_eval1)
        sol2 = solve_ivp(self.RGE, [log(1), log(self.phi_max)], [self.alpha_lambda0(), self.alpha_Y0()], t_eval=t_eval2)
        t_values = np.concatenate((sol1.t[::-1], sol2.t[1:]))
        return [t_values, sol1, sol2]

    def alpha_lambda(self, t):
        alpha_lambda_values = np.concatenate((self.sol1.y[0,::-1], self.sol2.y[0,1:]))
        return interp1d(self.t_values, alpha_lambda_values)(t)

    def alpha_Y(self, t):
        alpha_Y_values = np.concatenate((self.sol1.y[1,::-1], self.sol2.y[1,1:]))
        return interp1d(self.t_values, alpha_Y_values)(t)

    def t(self, phi, T=0):
        #phi, T = np.asanyarray((phi, T))
        return log(max(phi,T) / self.mu)
    
    # derivatives
    def dalpha_lambda(self, t):
        return derivative(self.alpha_lambda, t, 1e-6)
    def dalpha_Y(self, t):
        return derivative(self.alpha_Y, t, 1e-6)
    def dalpha_BL(self, t):
        return derivative(self.alpha_BL, t, 1e-6)
    def dt(self, phi, T):
        if phi <= T:
            return 0
        if phi > T:
            return 1 / phi
    
    # particles
    # def field_list(self):
    #     return [Field(1, self.m2_RHN1, 'fermion', 'right-handed neutrino'), \
    #             Field(3, self.m2_Zprime, 'boson', "Z'"), \
    #             Field(1, self.m2_Phi, 'boson', 'phi'), \
    #             Field(1, self.m2_G, 'boson', 'Goldstone boson')]
    def RHN1(self):
        return Field(1, self.m2_RHN1, self.dm2_RHN1, 'fermion', 'right-handed neutrino')
    def Zprime(self):
        return Field(3, self.m2_Zprime, self.dm2_Zprime, 'boson', "Z'")
    def Phi(self):
        return Field(1, self.m2_Phi, self.dm2_Phi, 'boson', 'phi')
    def G(self):
        return Field(1, self.m2_G, self.dm2_G, 'boson', 'Goldstone boson')

    # squared particle masses  
    def m2_RHN1(self, phi, T):  # right-handed neutrino
        return (4 * pi * self.alpha_Y( self.t(phi,T) )) * phi**2 / 2
    
    def m2_Zprime(self, phi, T): # B-L gauge boson
        return 4 * (4 * pi * self.alpha_BL( self.t(phi,T) )) * phi**2
    
    def m2_Phi(self, phi, T): # B-L symmetry-breaking scalar
        return 3 * (4 * pi * self.alpha_lambda( self.t(phi,T) )) * phi**2
    
    def m2_G(self, phi, T): # Goldstone boson
        return (4 * pi * self.alpha_lambda( self.t(phi,T) )) * phi**2
    
    # phi-derivatives of squared masses
    def dm2_RHN1(self, phi, T):
        dm2_dt = 2 * pi * self.dalpha_Y(self.t(phi, T)) * self.dt(phi, T) * phi**2 + \
            8 * pi * self.alpha_Y(self.t(phi, T)) * phi
        return dm2_dt
    def dm2_Zprime(self, phi, T):
        dm2_dt = 16 * pi * self.dalpha_BL(self.t(phi, T)) * self.dt(phi, T) * phi**2 + \
            32 * pi * self.alpha_BL(self.t(phi, T)) * phi
        return dm2_dt
    def dm2_Phi(self, phi, T):
        dm2_dt = 12 * pi * self.dalpha_lambda(self.t(phi, T)) * self.dt(phi, T) * phi**2 + \
            24 * pi * self.alpha_lambda(self.t(phi, T)) * phi
        return dm2_dt
    def dm2_G(self, phi, T):
        dm2_dt = 4 * pi * self.dalpha_lambda(self.t(phi, T)) * self.dt(phi, T) * phi**2 + \
            8 * pi * self.alpha_lambda(self.t(phi, T)) * phi
        return dm2_dt
    
      
    #Debeye masses?

    # potentials...only the real parts
    def Veff0(self, phi, T=0): # zero temperature RG-improved potential

        def negative_gamma(t): # anomalous dimension for one RHN
            return -(self.alpha_Y(t) - 24 * self.alpha_BL(t)) / 8 / pi
        def G(t):
            return exp( quad(negative_gamma, 0.0, t)[0] )
        
        def V0(phi):
            return pi * self.alpha_lambda(self.t(phi, T)) * G(self.t(phi, T))**4 * phi**4
      
        return np.real( V0(phi) - V0(self.phi_min) )

    def __call__(self, phi, T):
        if T==0:
            return self.Veff0(phi,T)
        else:
            def boson_sum(phi, T):
                return sum(field.dof * J_B(field.mass_squared(phi,T)/T**2) \
                        for field in self.field_list if field.type=='boson')
            def fermion_sum(phi, T):
                return sum(field.dof * J_F(field.mass_squared(phi,T)/T**2) \
                        for field in self.field_list if field.type=='fermion')

            def VT(phi):
                return self.Veff0(phi, T) + \
                        (T**4 / 2 / pi**2) * (boson_sum(phi,T) + fermion_sum(phi,T))

            return np.real( VT(phi) - VT(self.phi_min) )
        
    def dVeff(self, phi, T=0): # derivative with respect to phi
        def negative_gamma(t): # anomalous dimension for one RHN
            return -(self.alpha_Y(t) - 24 * self.alpha_BL(t)) / 8 / pi
        def G(t):
            return exp( quad(negative_gamma, 0.0, t)[0] )
        
        def dG4(t):
            return - ( self.alpha_Y(self.t(phi, T)) - 24 * self.alpha_BL(self.t(phi, T)) ) * G(self.t(phi, T))**4 / 2 / pi
        
        def dboson_sum(phi, T):
            return sum(field.dof * dJ_B(field.mass_squared(phi,T)/T**2) * field.dmass_squared(phi, T) \
                    for field in self.field_list if field.type=='boson')
        def dfermion_sum(phi, T):
            return sum(field.dof * dJ_F(field.mass_squared(phi,T)/T**2) * field.dmass_squared(phi, T) \
                    for field in self.field_list if field.type=='fermion')

        
        dV_dphi = 4 * pi * self.alpha_lambda(self.t(phi, T)) * G(self.t(phi, T))**4 * phi**3 + \
        pi * (self.dalpha_lambda(self.t(phi, T))) * G(self.t(phi, T))**4 * self.dt(phi, T) * phi**4 + \
        pi * self.alpha_lambda(self.t(phi, T)) * dG4(self.t(phi, T)) * self.dt(phi, T) * phi**4 + \
        (T**2 / 2 / pi**2) * (dboson_sum(phi, T) + dfermion_sum(phi, T))

        return np.real(dV_dphi)