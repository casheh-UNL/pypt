# parameter scan for first-order phase transitions

from .constants import *
from .pt_math import *

from .ftpot import *

from scipy.optimize import root
import random

class FOPTScan(object):
    def __init__(self, alpha_BL0_max=0.15, k_max=0.5, scan_points=1e2, temperature_range=[0.1, 1.0], phi_range=[0.1, 1.4], init=[1.0, 0.5]):
        self.alpha_BL0_max = alpha_BL0_max
        self.k_max = k_max

        self.scan_points = scan_points
        self.temperature_range = temperature_range
        self.phi_range = phi_range
        self.init = init

        self.results = []
    
    def roots(self, func):
        solutions = root(func, self.init).x
        return solutions
    
    def perform_scan(self):
        for _ in range(self.scan_points): # alternatively, use a while loop to find a given number of successful PTs
            alpha_BL0 = random.uniform(0, self.alpha_BL0_max)
            k = random.uniform(0, self.k_max)
            Veff = VeffBL(alpha_BL0=alpha_BL0, k=k)

            def V_dV(x):
                V = Veff(x[0], x[1])
                dV = Veff.dVeff(x[0], x[1])
                return [V, dV]
            
            result = self.roots(V_dV)
            self.results.append(result)

    def __call__(self):
        pass