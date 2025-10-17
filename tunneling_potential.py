# https://arxiv.org/pdf/1805.03680

import numpy as np
from scipy.integrate import solve_bvp, solve_ivp, quad, simpson
from scipy.special import gamma
from scipy.optimize import root_scalar, minimize_scalar

from pypt import *


def find_decimal_precision(val):
    val_str = repr(val)
    if '.' in val_str:
        decimal_part = val_str.split('.')[-1]
        return len(decimal_part.rstrip(')'))
    return 0

class TunnelingPotential:
    def __init__(self, V, dV, number_of_phis=200):
        """
        Initialize with a discrete set of phi values and the corresponding effective potential V(phi).
        
        Parameters:
        number_of_phis (float): Number of phi values to sample.
        V (function): Effective potential V(phi).
        """
        self.number_of_phis = number_of_phis

        if isinstance(V, VeffBL):
            self.phi_plus = V.phi_min
            self.phi_minus = V.find_minimum_phi # function of temperature
            self.V = V
            self.dV_dphi = dV
            self.d = 3
        else:
            self.phi_plus = 0 + 1e-6
            self.phi_minus = lambda T: 1
            self.V = V
            self.dV_dphi = dV
            self.d = 4
            self.phi_range = np.linspace(self.phi_plus, self.phi_minus(T=None), number_of_phis)

        self.tol = 1e-14
        self.data = []
    
    def tunneling_potential_ODEs(self, phi, V_t, p):
        """
        Differential equation (34) for the tunneling potential V_t(phi).
        
        Parameters:
        phi (float): Field value.
        V_t (float): Tunneling potential at phi.
        dV_t_dphi (float): First derivative of tunneling potential at phi.
        
        Returns:
        float: Second derivative of tunneling potential at phi.
        """
        Vt, dVt = V_t
        V, dV = self.V(phi), self.dV_dphi(phi)
        dVt_dphi = dVt
        # d2Vt_dphi2 = ((dV * dVt) - (self.d/(self.d-1))*np.power(dVt, 2)) / (2 * (Vt - V))
        num = (dV * dVt) - (self.d/(self.d-1))*np.power(dVt, 2)
        den = 2 * (Vt - V) * (-1) ### POSSIBLE TYPO IN PAPER
        # let 0/0 = 0 to avoid division by zero
        d2Vt_dphi2 = np.divide(num, den, out=np.zeros_like(num, dtype=float), where=den != 0)
        return np.vstack((dVt_dphi, d2Vt_dphi2))
    
    def jacobian(self, phi, Vt_dVt, p):
        Vt, dVt = Vt_dVt
        V, dV = self.V(phi), self.dV_dphi(phi)
        D = self.d / (self.d-1)
        m = len(phi)

        J = np.zeros((2,2,m))
        J[0,0,:] = 0
        J[0,1,:] = 1
        J[1,0,:] = (dV*dVt - D*dVt**2) / (2*(V-Vt)**2) # singular Jacobian at boundaries where V = Vt
        J[1,1,:] = (dV - 2*D*dVt) / (2*(V-Vt))

        Jp = np.zeros((2,1,m))

        return J, Jp

    # boundary conditions (use p[0] as φ₀)
    def boundary_conditions(self, V_ta, V_tb, p):
        """
        Boundary conditions for the tunneling potential V_t(phi) based on equations (12) and (13).
        
        Parameters:
        V_ta (float): V_t at the metastable phi (false vacuum).
        V_tb (float): V_t at the global minimum phi (true vacuum).
        
        Returns:
        tuple: Residuals for the boundary conditions.
        """
        phi0 = p[0]
        return np.array([
            # V_ta[0] - self.V(self.phi_plus),                       # V_t(φ+) = V(φ+)
            V_ta[1] - self.dV_dphi(self.phi_plus),                 # V_tp(φ+) = dV_dphi(φ+)
            V_tb[0] - self.V(phi0),                                # V_t(φ₀) = V(φ₀)
            V_tb[1] - ((self.d-1)/self.d) * self.dV_dphi(phi0)     # V_tp(φ₀) = ((d-1)/d) dV/dφ(φ₀)
        ])
    
    def solve_tunneling_potential(self, T=None):
        """
        Solves for the tunneling potential V_t(phi) in the boundary-value problem.

        Returns:
        array: Values of the tunneling potential V_t corresponding to self.phi_range.
        """
        # initial guesses
        phi_guess = self.phi_range
        phi0_guess = np.array([0.95 * self.phi_minus(T)])
        # Vt_guess, dVt_guess = 0.95*self.V(phi_guess), self.dV_dphi(phi_guess)
        Vt_guess, dVt_guess = self.V_t4_(phi_guess, phi0_guess), self.dV_t4(phi_guess, phi0_guess)

        # solve the BVP
        sol = solve_bvp(self.tunneling_potential_ODEs,
                        self.boundary_conditions,
                        x=phi_guess, y=(Vt_guess, dVt_guess), p=phi0_guess,
                        fun_jac=self.jacobian
                        )

        # if sol.status != 0:
        #     raise RuntimeError("BVP solver did not converge")
        
        self.phi_0 = sol.p[0]
        self.V_t, self.dV_t = sol.sol(self.phi_range)
        return sol
    
    ############################
    # EUCLIDEAN/BOUNCE ACTIONS #
    ############################
    def compute_action(self):
        """
        Computes the Euclidean bounce action S_E using equation (37).
        
        Returns:
        float: The computed action S_E.
        """
        if not hasattr(self, 'V_t'):
            raise RuntimeError("Tunneling potential V_t not computed. Call solve_tunneling_potential() first.")
        
        # integral range
        
        V, V_t, dV_t = self.V(self.phi_range), self.V_t, self.dV_t

        coeff = np.power(self.d-1, self.d-1) * np.power(2*np.pi, self.d/2) / gamma(1+self.d/2)
        integrand = np.power(V - V_t, self.d/2) / np.power(abs(dV_t), self.d-1)
        S_E = coeff * simpson(integrand, self.phi_range)
        return S_E
    
    def S_E(self, phi0, V_t=None, dV_t=None, T=None, phi_dist='hybrid', transition=0.1):
        if (V_t is None) and (dV_t is None):
            def V_tun(phi): return self.V_t4(phi, phi0, T)
            def dV_tun(phi): return self.dV_t4(phi, phi0, T)
        else:
            def V_tun(phi): return V_t(phi, phi0, T)
            def dV_tun(phi): return dV_t(phi, phi0, T)

        if phi_dist == 'logarithmic':
            phi_range = np.linspace(np.log(self.phi_plus), np.log(phi0), self.number_of_phis)[1:]
            V, Vt, dVt = self.V(np.exp(phi_range), T), V_tun(np.exp(phi_range)), dV_tun(np.exp(phi_range))

            num, den = np.power(V-Vt, self.d/2), np.power(abs(dVt), self.d-1)
            integrand = num / den * np.exp(phi_range)
        elif phi_dist == 'linear':
            phi_range = np.linspace(self.phi_plus, phi0, self.number_of_phis)[1:]
            # leaving out phi_plus since V - V_t = 0 and dV_t = 0 there
            V, Vt, dVt = self.V(phi_range, T), V_tun(phi_range), dV_tun(phi_range)

            num = np.power(V - Vt, self.d/2)
            den = np.power(abs(dVt), self.d-1)
            integrand = num / den
        elif phi_dist == 'hybrid':
            transition_phi = self.phi_plus + transition*(phi0 - self.phi_plus)
            num_points_log, num_points_lin = int(self.number_of_phis/2), int(self.number_of_phis/2)

            log_phis = np.logspace(np.log10(self.phi_plus), np.log10(transition_phi), num_points_log, base=10.0)
            lin_phis = np.linspace(transition_phi, phi0, num_points_lin)

            phi_range = np.unique(np.concatenate((log_phis, lin_phis)))[1:]
            V, Vt, dVt = self.V(phi_range, T), V_tun(phi_range), dV_tun(phi_range)
            num = np.power(V - Vt, self.d/2)
            den = np.power(abs(dVt), self.d-1)
            integrand = num / den
        else:
            raise Exception("phi_dist must be 'linear', 'logarithmic', or 'hybrid'")
        # note: if any value in integrand is np.nan, simpson returns np.nan for the integral

        coeff = np.power(self.d-1, self.d-1) * np.power(2*np.pi, self.d/2) / gamma(1+self.d/2)
        S_E = coeff * simpson(y=integrand, x=phi_range)
        return S_E
    
    def S_E_quad(self, phi0, V_t=None, dV_t=None, T=None, epsilon=1e-6):
        def integrand(phi, phi0, V_t, dV_t, T):
            V = self.V(phi, T)
            if (V_t is None) and (dV_t is None):
                Vt, dVt = self.V_t4(phi, phi0, T), self.dV_t4(phi, phi0, T)
            else:
                Vt, dVt = V_t(phi, phi0, T), dV_t(phi, phi0, T)

            num = np.power(V - Vt, self.d/2)
            den = np.power(abs(dVt), self.d-1)
            return num / den

        coeff = np.power(self.d-1, self.d-1) * np.power(2*np.pi, self.d/2) / gamma(1+self.d/2)
        
        S_E = coeff * quad(integrand, self.phi_plus+epsilon, phi0-epsilon, args=(phi0, V_t, dV_t, T))[0]
        return S_E
    
    ##############
    # FINDING φ0 #
    ##############
    def find_phi0_and_action(self, T=None, V_t=None, dV_t=None, phi_dist='hybrid', transition_phi=0.1, max_iterations=20, tol=1e-5, shift=0.5):
        """
        Calculates phi0 by finding the value at which the Euclidean/bounce action is minimized.
        By default, the fourth-order approximation V_t4 is used for the tunneling potential.
        """
        def safe_action(phi0):
            val = self.S_E(phi0, V_t, dV_t, T, phi_dist, transition_phi)
            return val if np.isfinite(val) else np.inf
        
        phi0_low, phi0_high = self.phi_plus, self.phi_minus(T)
        if phi0_low >= phi0_high:
            raise ValueError(f'True/broken VEV is lower than false/symmetric VEV. \n T = {T} \n phi_plus = {phi0_low} \n phi_minus = {phi0_high}')
        iteration = 0
        while iteration <= max_iterations:

            # if phi0_high - phi0_low < 0:
            #     break # avoid invalid bounds
            iteration += 1

            res = minimize_scalar(safe_action, bounds=(phi0_low, phi0_high))
            if res.success:
                if (res.fun == np.inf) and (abs(res.x - phi0_high) < tol):
                    # res.fun = np.inf usually indicates the solver found phi0_high to be the minimum
                    phi0_high -= shift*(phi0_high - phi0_low)
                # in case the bounds don't contain the true minimum:
                elif (abs(res.x - phi0_low) < tol) and (phi0_low != self.phi_plus):
                    phi0_low += shift*(phi0_high - phi0_low)
                elif (abs(res.x - phi0_high) < tol) and (phi0_high != self.phi_minus(T)):
                    phi0_high -= shift*(phi0_high - phi0_low)
                else:
                    return (res.x, res.fun)
            else:
                print(f'Warning: refined minimization failed ({res.message}). Returning coarse grid minimum.')
                break
        
        print('Warning: maximum iterations reached without successful minimization.')
        return (np.nan, np.inf)
        
    def find_phi0_and_action_GRID(self, T=None, V_t=None, dV_t=None, grid_points=101, refine_factor=0.1, phi_dist='hybrid', phi0_dist='hybrid', transition_phi=1., transition_phi0=1.):
        """
        Calculates phi0 by finding the value at which the Euclidean/bounce action is minimized.
        By default, the fourth-order approximation V_t4 is used for the tunneling potential.
        This method could be quicker for temperatures below about 1% of the critical temperature.
        """
        action = lambda phi0: self.S_E(phi0, V_t, dV_t, T, phi_dist, transition_phi)
        
        # grid search with non-np.nan action values
        if phi0_dist == 'logarithmic':
            phi0_grid = np.logspace(np.log(self.phi_plus), np.log(self.phi_minus(T)), grid_points, base=np.e)[1:]
        elif phi0_dist == 'linear':
            phi0_grid = np.linspace(self.phi_plus, self.phi_minus(T), grid_points)[1:]
        elif phi0_dist == 'hybrid':
            transition_phi = self.phi_plus + transition_phi0*(self.phi_minus(T) - self.phi_plus)
            num_points_log, num_points_lin = int(grid_points/2), int(grid_points/2)

            log_phis = np.logspace(np.log10(self.phi_plus), np.log10(transition_phi), num_points_log, base=10.0)
            lin_phis = np.linspace(transition_phi, self.phi_minus(T), num_points_lin)

            phi0_grid = np.unique(np.concatenate((log_phis, lin_phis)))[1:]
        else:
            raise Exception("phi_dist must be 'linear', 'logarithmic', or 'hybrid'")
        
        action_vals = np.array([action(phi0) for phi0 in phi0_grid])
        finite_mask = np.isfinite(action_vals)
        min_action_index = np.argmin(action_vals[finite_mask])
        min_phi0_coarse = phi0_grid[finite_mask][min_action_index]

        # refine the coarse search above
        window_width = (self.phi_minus(T) - self.phi_plus) * refine_factor
        phi0_low = max(self.phi_plus, min_phi0_coarse - window_width/2.)
        phi0_high = min(self.phi_minus(T), min_phi0_coarse + window_width/2.)
        # phi0_low = min(phi0_grid[finite_mask])
        # phi0_high = max(phi0_grid[finite_mask])
        # print(phi0_low, phi0_high)
        
        # return (min_phi0_coarse, action_vals[finite_mask][min_action_index])
        res = minimize_scalar(action, bounds=(phi0_low, phi0_high), method='bounded')
        if res.success:
            return (res.x, res.fun)
        else:
            print(f'Warning: refined minimization failed ({res.message}). Returning coarse grid minimum.')
            return (min_phi0_coarse, action_vals[finite_mask][min_action_index])

    ######################################
    # TUNNELING POTENTIAL APPROXIMATIONS #
    ######################################
    def V_t1_(self, phi, phi0, T=None):
        return self.V(phi0, T) * phi/phi0
    def V_t2_(self, phi, phi0, T=None):
        V0, dV0 = self.V(phi0, T), self.dV_dphi(phi0, T)
        D = (self.d-1)/self.d
        return self.V_t1_(phi, phi0, T) + (phi/phi0)*(D*dV0 - V0/phi0)*(phi-phi0)
    def V_t3_(self, phi, phi0, T=None):
        V0, dV0 = self.V(phi0, T), self.dV_dphi(phi0, T)
        D = (self.d-1)/self.d
        return self.V_t2_(phi, phi0, T) + (phi/phi0)*(D*dV0/phi0 - 2*V0/phi0**2)*(phi-phi0)**2
    def V_t4_(self, phi, phi0, T=None):
        a4 = self.a4(phi0, T)
        return self.V_t3_(phi, phi0, T) + a4*phi**2 * (phi - phi0)**2
    
    def dV_t1_(self, phi, phi0, T=None):
        return self.V(phi0, T) / phi0
    def dV_t2_(self, phi, phi0, T=None):
        V0, dV0 = self.V(phi0, T), self.dV_dphi(phi0, T)
        D = (self.d-1)/self.d
        return self.dV_t1_(phi, phi0, T) + (1/phi0)*(D*dV0 - V0/phi0) * (2*phi - phi0)
    def dV_t3_(self, phi, phi0, T=None):
        V0, dV0 = self.V(phi0, T), self.dV_dphi(phi0, T)
        D = (self.d-1)/self.d
        return self.dV_t2_(phi, phi0, T) + (1/phi0)*(D*dV0/phi0 - 2*V0/phi0**2) * (3*phi**2 - 4*phi0*phi + phi0**2)
    def dV_t4_(self, phi, phi0, T=None):
        a4 = self.a4(phi0, T)
        return self.dV_t3_(phi, phi0, T) + a4*(4*phi**3 - 6*phi0*phi**2 + 2*phi0**2 * phi)
    
    def d2V_t1_(self, phi, phi0, T=None):
        return 0
    def d2V_t2_(self, phi, phi0, T=None):
        V0, dV0 = self.V(phi0, T), self.dV_dphi(phi0, T)
        D = (self.d-1)/self.d
        return self.d2V_t1_(phi, phi0, T) + (2/phi0)*(D*dV0 - V0/phi0)
    def d2V_t3_(self, phi, phi0, T=None):
        V0, dV0 = self.V(phi0, T), self.dV_dphi(phi0, T)
        D = (self.d-1)/self.d
        return self.d2V_t2_(phi, phi0, T) + (1/phi0)*(D*dV0/phi0 - 2*V0/phi0**2) * (6*phi - 4*phi0)
    
    def a4(self, phi0, T=None):
        # phi at which V is maximized (top of barrier)
        phiT = minimize_scalar(lambda phi: -self.V(phi, T), bounds=(self.phi_plus, self.phi_minus(T)), method='bounded').x

        VT, dVT = self.V(phiT, T), 0
        V_t3T, dV_t3T, d2V_t3T = self.V_t3_(phiT, phi0, T), self.dV_t3_(phiT, phi0, T), self.d2V_t3_(phiT, phi0, T)
        D = (self.d-1)/self.d
        
        A = -4*phiT**2 * (phi0-phiT)**2 * ((D-1)*phi0**2 + (4-6*D)*phi0*phiT + (6*D-4)*phiT**2)
        B = -2*D*dVT*phiT * (phi0**2 - 3*phi0*phiT + 2*phiT**2) + 4*D*(phi0**2 - 6*phi0*phiT + 6*phiT**2)*(VT-V_t3T) \
            -2*D*phiT**2 * (phi0-phiT)**2 * d2V_t3T + 4*phiT*(phi0**2 - 3*phi0*phiT + 2*phiT**2)*dV_t3T
        C = -D*dVT*dV_t3T + 2*D*(VT-V_t3T)*d2V_t3T + dV_t3T**2
        
        return (-B - np.sqrt(B*B - 4*A*C)) / (2*A)
    
    # To restrict freedom on phi0, we ensure V_t(phi, phi0) <= V(phi) and dV_t <= 0.
    # ...this ruins the solve_bvp solver
    # VeffBL has a working precision of 18 or 19 digits
    def V_t1(self, phi, phi0, T=None):
        V, Vt1, dVt1 = self.V(phi, T), self.V_t1_(phi, phi0, T), self.dV_t1_(phi, phi0, T)
        return np.where(np.abs(Vt1 - V) < self.tol, V, np.where((Vt1 > V) | (dVt1 > 0), np.nan, Vt1))
    def V_t2(self, phi, phi0, T=None):
        V, Vt2, dVt2 = self.V(phi, T), self.V_t2_(phi, phi0, T), self.dV_t2_(phi, phi0, T)
        return np.where(np.abs(Vt2 - V) < self.tol, V, np.where((Vt2 > V) | (dVt2 > 0), np.nan, Vt2))
    def V_t3(self, phi, phi0, T=None):
        V, Vt3, dVt3 = self.V(phi, T), self.V_t3_(phi, phi0, T), self.dV_t3_(phi, phi0, T)
        return np.where(np.abs(Vt3 - V) < self.tol, V, np.where((Vt3 > V) | (dVt3 > 0), np.nan, Vt3))
    def V_t4(self, phi, phi0, T=None):
        V, Vt4, dVt4 = self.V(phi, T), self.V_t4_(phi, phi0, T), self.dV_t4_(phi, phi0, T)
        return np.where(np.abs(Vt4 - V) < self.tol, V, np.where((Vt4 > V) | (dVt4 > 0), np.nan, Vt4))
        # return np.where((Vt4 > V) | (dVt4 > 0), np.nan, Vt4)
    
    def dV_t1(self, phi, phi0, T=None):
        V, Vt1, dVt1 = self.V(phi, T), self.V_t1_(phi, phi0, T), self.dV_t1_(phi, phi0, T)
        return np.where(np.abs(Vt1 - V) < self.tol, dVt1, np.where((Vt1 > V) | (dVt1 > 0), np.nan, dVt1))
    def dV_t2(self, phi, phi0, T=None):
        V, Vt2, dVt2 = self.V(phi, T), self.V_t2_(phi, phi0, T), self.dV_t2_(phi, phi0, T)
        return np.where(np.abs(Vt2 - V) < self.tol, dVt2, np.where((Vt2 > V) | (dVt2 > 0), np.nan, dVt2))
    def dV_t3(self, phi, phi0, T=None):
        V, Vt3, dVt3 = self.V(phi, T), self.V_t3_(phi, phi0, T), self.dV_t3_(phi, phi0, T)
        return np.where(np.abs(Vt3 - V) < self.tol, dVt3, np.where((Vt3 > V) | (dVt3 > 0), np.nan, dVt3))
    def dV_t4(self, phi, phi0, T=None):
        V, Vt4, dVt4 = self.V(phi, T), self.V_t4_(phi, phi0, T), self.dV_t4_(phi, phi0, T)
        return np.where(np.abs(Vt4 - V) < self.tol, dVt4, np.where((Vt4 > V) | (dVt4 > 0), np.nan, dVt4))
        # return np.where((Vt4 > V) | (dVt4 > 0), np.nan, dVt4)
    
    def d2V_t1(self, phi, phi0, T=None):
        V, Vt1 = self.V(phi, T), self.V_t1_(phi, phi0, T)
        d2Vt1 = 0
        return np.where(np.abs(Vt1 - V) < self.tol, d2Vt1, np.where(Vt1 > V, np.nan, d2Vt1))
    def d2V_t2(self, phi, phi0, T=None):
        V, Vt2 = self.V(phi, T), self.V_t2_(phi, phi0, T)
        d2Vt2 = self.d2V_t2_(phi, phi0, T)
        return np.where(np.abs(Vt2 - V) < self.tol, d2Vt2, np.where(Vt2 > V, np.nan, d2Vt2))
    def d2V_t3(self, phi, phi0, T=None):
        V, Vt3 = self.V(phi, T), self.V_t3_(phi, phi0, T)
        d2Vt3 = self.d2V_t3_(phi, phi0, T)
        return np.where(np.abs(Vt3 - V) < self.tol, d2Vt3, np.where(Vt3 > V, np.nan, d2Vt3))
        # return np.where(Vt3 > V, np.nan, d2Vt3)

    
    ###################
    # SHOOTING METHOD #
    ###################

    def tunneling_potential_ODEs_shooting(self, phi, V_t):
        """
        Differential equation (34) for the tunneling potential V_t(phi).
        
        Parameters:
        phi (float): Field value.
        V_t (float): Tunneling potential at phi.
        dV_t_dphi (float): First derivative of tunneling potential at phi.
        
        Returns:
        float: Second derivative of tunneling potential at phi.
        """
        Vt, dVt = V_t
        V, dV = self.V(phi), self.dV_dphi(phi)
        dVt_dphi = dVt
        # d2Vt_dphi2 = ((dV * dVt) - (self.d/(self.d-1))*np.power(dVt, 2)) / (2 * (Vt - V))
        num = (dV * dVt) - (self.d/(self.d-1))*np.power(dVt, 2)
        den = 2 * (Vt - V) * (-1)
        # let 0/0 = 0 to avoid division by zero
        d2Vt_dphi2 = np.divide(num, den, out=np.zeros_like(num, dtype=float), where=den != 0)
        return np.array([dVt_dphi, d2Vt_dphi2])
    
    def objective_function_(self, phi0):
        """
        Objective function to minimize: the difference between the calculated
        dVt/dphi at phi0 and the required boundary condition value of 
        dVt/dphi(phi0) = (d-1)/d * dV/dphi(phi0)
        """
        # "initial" conditions
        Vt_phi0, dVt_phi0 = self.V(phi0), ((self.d-1)/self.d) * self.dV_dphi(phi0)
        
        # t_eval_reverse = np.linspace(phi0, self.phi_plus, 10)
        sol = solve_ivp(self.tunneling_potential_ODEs_shooting, t_span=(phi0, self.phi_plus), # notice the reversed ordering
                        y0=(Vt_phi0, dVt_phi0),
                        # t_eval=t_eval_reverse
                        )
    
        if sol.status != 0:
            # self.data.append([sol, phi0])
            return np.inf # indicate failure
        
        Vt, dVt = sol.y
        # self.data.append([f'phi0 = {phi0}, Vt(phi+) = {Vt[-1]}, Vt(phi0) = {Vt[0]}'])

        # boundary conditions
        Vt_phi_plus, dVt_phi_plus = self.V(self.phi_plus), self.dV_dphi(self.phi_plus)
        # return Vt[-1] - Vt_phi_plus
        return dVt[-1] - dVt_phi_plus
    
    def objective_function(self, phi0):
        """
        Objective function to minimize: the difference between the calculated
        dVt/dphi at phi0 and the required boundary condition value of 
        dVt/dphi(phi0) = (d-1)/d * dV/dphi(phi0)

        Page 7 of https://arxiv.org/pdf/2404.19657 recommends starting from phi_plus
        """
        # "initial" conditions
        Vt_phi_plus, dVt_phi_plus = self.V(self.phi_plus), self.dV_dphi(self.phi_plus)
        
        # t_eval_reverse = np.linspace(phi0, self.phi_plus, 10)
        sol = solve_ivp(self.tunneling_potential_ODEs_shooting, t_span=(self.phi_plus, phi0),
                        y0=(Vt_phi_plus, dVt_phi_plus),
                        # t_eval=t_eval_reverse
                        )
    
        if sol.status != 0:
            # self.data.append([sol, phi0])
            return np.inf # indicate failure
        
        Vt, dVt = sol.y
        # self.data.append([f'phi0 = {phi0}, Vt(phi+) = {Vt[-1]}, Vt(phi0) = {Vt[0]}'])

        # boundary conditions
        Vt_phi0, dVt_phi0 = self.V(phi0), ((self.d-1)/self.d) * self.dV_dphi(phi0)
        # return Vt[-1] - Vt_phi0
        return dVt[-1] - dVt_phi0
    
    def find_tunneling_potential_shooting(self, r=0.9):
        phi0_guess = r * self.phi_minus(T=None)
        try:
            # result = root_scalar(self.objective_function, x0=self.phi_minus-2e-1, x1=phi0_guess)
            result = root_scalar(self.objective_function, bracket=[phi0_guess, 1.1*self.phi_minus(T=None)], method='brentq')
            if result.converged:
                phi0_solution = result.root
                return phi0_solution
            else:
                print("root-finding failed:", result.flag)
        except Exception as e:
            print(f'Integration failed to find phi0: {e}')
            return None


    def objective_function_vec(self, phi0_array):
        """
        Objective function to minimize: the difference between the calculated
        dVt/dphi at phi0 and the required boundary condition value of 
        dVt/dphi(phi0) = (d-1)/d * dV/dphi(phi0)
        """
        phi0 = phi0_array[0]
        # "initial" conditions
        Vt_phi0, dVt_phi0 = self.V(phi0), ((self.d-1)/self.d) * self.dV_dphi(phi0)
        
        # t_eval_reverse = np.linspace(phi0, self.phi_plus, 100)
        sol = solve_ivp(self.tunneling_potential_ODEs_shooting, t_span=(phi0, self.phi_plus), # notice the reversed ordering
                        y0=(Vt_phi0, dVt_phi0),
                        # t_eval=t_eval_reverse
                        )
    
        if sol.status != 0:
            # self.data.append([sol, phi0])
            return np.array([
                # np.inf,
                np.inf
                ])  # indicate failure
        
        Vt, dVt = sol.y
        # self.data.append([f'phi0 = {phi0}'])

        # boundary conditions
        Vt_phi_plus, dVt_phi_plus = self.V(self.phi_plus), self.dV_dphi(self.phi_plus)
        return np.array([
            # Vt[-1] - Vt_phi_plus,
            dVt[-1] - dVt_phi_plus
        ])

    def find_tunneling_potential_shooting_vec(self, r=0.9):
        phi0_guess = np.array([r * self.phi_minus(T=None)])
        try:
            result = root(self.objective_function_vec, phi0_guess)
            if result.success:
                phi0_solution = result.x[0]
                print('phi0 found:', phi0_solution)
            else:
                print("root-finding failed:", result.message)
        except Exception as e:
            print(f'Integration failed to find phi0: {e}')
            return None, None




    def shooting_potential_ivp(self, phi, V_t_prime, phi0):
        """
        First-order ODE for dVt/dphi based on equation (36) with a fixed phi0.

        Parameters:
        phi (float): Field value.
        V_t_prime (float): First derivative of tunneling potential at phi.
        phi0 (float): The assumed value of phi0.

        Returns:
        float: Second derivative of tunneling potential at phi.
        """
        V_val, dV_val = self.V(phi), self.dV_dphi(phi)
        # approximate Vt by integrating Vt'
        Vt_val = self.V(self.phi_plus) + simpson(V_t_prime, self.phi_shoot[:np.searchsorted(self.phi_shoot, phi)])

        num = (dV_val * V_t_prime) - (self.d/(self.d-1))*Vt_val
        den = 2 * (Vt_val - V_val) * (-1)
        d2Vt_dphi2 = np.divide(num, den, out=np.zeros_like(num, dtype=float), where=den != 0)
        return d2Vt_dphi2

    def solve_tunneling_potential_shoot(self, phi0_guess, dVt_plus_guess=None):
        """
        Solves for the tunneling potential V_t(phi) using the shooting method.

        Parameters:
        phi0_guess (float): Initial guess for phi0.
        dVt_plus_guess (float, optional): Initial guess for dVt/dphi at phi_plus.
                                                If None, defaults to dV/dphi at phi_plus.

        Returns:
        tuple: (phi0_solution, Vt_solution) if a solution is found, otherwise (None, None).
        """
        if dVt_plus_guess is None:
            dVt_plus_guess = self.dV_dphi(self.phi_plus)

        def objective_function(phi0):
            """
            Objective function to minimize: the difference between the calculated
            dVt/dphi at phi0 and the required boundary condition value of 
            dVt/dphi(phi0) = (d-1)/d * dV/dphi(phi0)
            """
            def ode_system(phi, y):
                dVt = y[0]
                d2Vt = self.shooting_potential_ivp(phi, dVt, phi0)
                return [d2Vt]

            # Solve the IVP for dVt/dphi
            sol = solve_ivp(ode_system, (self.phi_plus, phi0), [dVt_plus_guess],
                            dense_output=True, t_eval=self.phi_shoot[self.phi_shoot <= phi0],
                            max_step=0.01)

            if sol.status != 0:
                return np.inf  # indicate failure

            dVt_at_phi0 = sol.y[0, -1]
            target_dVt_at_phi0 = ((self.d - 1) / self.d) * self.dV_dphi(phi0)
            return dVt_at_phi0 - target_dVt_at_phi0

        # Find the root of the objective function to determine phi0
        try:
            phi0_solution = root_scalar(objective_function, x0=self.phi_minus-1e-1, x1=self.phi_minus*0.9)
            self.phi_0_shoot = phi0_solution

            # Once phi0 is found, solve for Vt with that phi0
            def final_ode_system(phi, y):
                dVt = y[0]
                d2Vt = self.shooting_potential_ivp(phi, dVt, self.phi_0_shoot)
                return [d2Vt]

            final_sol = solve_ivp(final_ode_system, (self.phi_plus, self.phi_0_shoot), [dVt_plus_guess],
                                  dense_output=True, t_eval=np.linspace(self.phi_plus, self.phi_0_shoot, self.integration_points),
                                  max_step=0.01)

            if final_sol.status == 0:
                dVt_solution = final_sol.y[0]
                # Approximate Vt by integrating Vt'
                vt_solution = np.array([self.V(self.phi_plus) + simpson(dVt_solution[:i+1], np.linspace(self.phi_plus, final_sol.t[i], i+1))
                                        for i in range(len(final_sol.t))])
                self.Vt_shoot = vt_solution
                return self.phi_0_shoot, self.Vt_shoot, final_sol.t
            else:
                return None, None, None

        except ValueError:
            print("Error: Could not find a root for phi0 within the given bounds.")
            return None, None, None
        except Exception as e:
            print(f"An error occurred during the shooting method: {e}")
            return None, None, None