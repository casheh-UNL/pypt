# Math imports


from .constants import *


import numpy as np
from numpy import sqrt, log, exp, log10, power, sin, cos, tan, heaviside
import cmath

from scipy.optimize import fsolve, root, root_scalar
from scipy.integrate import quad, solve_ivp, simpson
# from scipy.misc import derivative # outdated
from numdifftools import Derivative as derivative
from scipy.interpolate import interp1d
from scipy.special import kv
from scipy.signal import argrelmin

    
# thermal integrals using scipy.integrate.quad return the real part with a warning about complex values
# alternatively, thermal integrals package: https://github.com/andrewfowlie/thermal_funcs

n_list = range(1,11)

def J_B(a):
    # def integrand(a):
    #     return lambda x: np.real( x**2 * log(1 - exp(-cmath.sqrt(x**2 + a))) )
    # try:
    #     result = quad(integrand(a), 0.0, np.inf)[0]
    # except:
    #     result = np.array( [quad(integrand(a_), 0.0, np.inf)[0] for a_ in a]).reshape(np.shape(a))
    # approximation

    try:
        result = -a * sum( (1)**n * kv(2, n * cmath.sqrt(a)) / n**2 for n in n_list )
    except:
        result = []
        for a_ in a:
            result.append( -a_ * sum( (1)**n * kv(2, n * cmath.sqrt(a_)) / n**2 for n in n_list ) )
        result = np.array(result).reshape(np.shape(a))
    return result

def J_F(a):
    # def integrand(a):
    #     return lambda x: np.real( x**2 * log(1 + exp(-cmath.sqrt(x**2 + a))) )
    # try:
    #     result = quad(integrand(a), 0.0, np.inf)[0]
    # except:
    #     result = np.array( [quad(integrand(a_), 0.0, np.inf)[0] for a_ in a]).reshape(np.shape(a))
    # approximation

    try:
        result = -a * sum( (-1)**n * kv(2, n * cmath.sqrt(a)) / n**2 for n in n_list )
    except:
        result = []
        for a_ in a:
            result.append( -a_ * sum( (-1)**n * kv(2, n * cmath.sqrt(a_)) / n**2 for n in n_list ) )
        result = np.array(result).reshape(np.shape(a))
    return result

def dJ_B(a):
    #integrand = lambda x: np.real( exp(-cmath.sqrt(a + x**2)) * x**2 / \
    #            ( 2 * (1 - exp(-cmath.sqrt(a + x**2))) * cmath.sqrt(a + x**2) ) )
    #result, error = quad(integrand, 0.0, np.inf)
    # approximation

    try:
        result = 0.25 * sum( ((1)**n / n**2) * (n*cmath.sqrt(a)*kv(1, n * cmath.sqrt(a)) - 4*kv(2, n * cmath.sqrt(a)) + n*cmath.sqrt(a)*kv(3, n * cmath.sqrt(a))) for n in n_list )
    except:
        result = []
        for a_ in a:
            result.append( 0.25 * sum( ((1)**n / n**2) * (n*cmath.sqrt(a_)*kv(1, n * cmath.sqrt(a_)) - 4*kv(2, n * cmath.sqrt(a_)) + n*cmath.sqrt(a_)*kv(3, n * cmath.sqrt(a_))) for n in n_list ) )
    return np.array(result).reshape(np.shape(a))

def dJ_F(a):
    #integrand = lambda x: np.real( -exp(-cmath.sqrt(a + x**2)) * x**2 / \
    #            ( 2 * (1 + exp(-cmath.sqrt(a + x**2))) * cmath.sqrt(a + x**2) ) )
    #result, error = quad(integrand, 0.0, np.inf)
    # approximation
    
    try:
        result = 0.25 * sum( ((-1)**n / n**2) * (n*cmath.sqrt(a)*kv(1, n * cmath.sqrt(a)) - 4*kv(2, n * cmath.sqrt(a)) + n*cmath.sqrt(a)*kv(3, n * cmath.sqrt(a))) for n in n_list )
    except:
        result = []
        for a_ in a:
            result.append( 0.25 * sum( ((-1)**n / n**2) * (n*cmath.sqrt(a_)*kv(1, n * cmath.sqrt(a_)) - 4*kv(2, n * cmath.sqrt(a_)) + n*cmath.sqrt(a_)*kv(3, n * cmath.sqrt(a_))) for n in n_list ) )
    return np.array(result).reshape(np.shape(a))