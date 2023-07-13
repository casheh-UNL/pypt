# Math imports


from .constants import *


import numpy as np
from numpy import sqrt, log, exp, log10, power, sin, cos, tan
import cmath

from scipy.optimize import fsolve
from scipy.integrate import quad, solve_ivp
from scipy.misc import derivative
from scipy.interpolate import interp1d
from scipy.special import kv

    
# thermal integrals using scipy.integrate.quad return the real part with a warning about complex values
# alternatively, thermal integrals package: https://github.com/andrewfowlie/thermal_funcs

def J_B(a):
    integrand = lambda x: x**2 * log(1 - exp(-cmath.sqrt(x**2 + a)))
    result, error = quad(integrand, 0.0, np.inf)
    # approximation
    #n_list = range(1,11)
    #result = -a * sum( (1)**n * kv(2, n * cmath.sqrt(a)) / n**2 for n in n_list )
    return result

def J_F(a):
    integrand = lambda x: x**2 * log(1 + exp(-cmath.sqrt(x**2 + a)))
    result, error = quad(integrand, 0.0, np.inf)
    # approximation
    #n_list = range(1,11)
    #result = -a * sum( (-1)**n * kv(2, n * cmath.sqrt(a)) / n**2 for n in n_list )
    return result

def dJ_B(a):
    integrand = lambda x: exp(-cmath.sqrt(a + x**2)) * x**2 / \
                ( 2 * (1 - exp(-cmath.sqrt(a + x**2))) * cmath.sqrt(a + x**2))
    result, error = quad(integrand, 0, np.inf)
    # approximation
    #n_list = range(1,11)
    #result = 0.5 * cmath.sqrt(a) * sum( (1)**n * kv(1, n * cmath.sqrt(a)) / n for n in n_list )
    return result

def dJ_F(a):
    integrand = lambda x: -exp(-cmath.sqrt(a + x**2)) * x**2 / \
                ( 2 * (1 + exp(-cmath.sqrt(a + x**2))) * cmath.sqrt(a + x**2))
    result, error = quad(integrand, 0, np.inf)
    # approximation
    #n_list = range(1,11)
    #result = 0.5 * cmath.sqrt(a) * sum( (-1)**n * kv(1, n * cmath.sqrt(a)) / n for n in n_list )
    return result