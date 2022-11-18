import enterprise.signals.parameter as parameter
import src.models_utils as aux
from scipy.special import gamma
import natpy as nat
import numpy as np

name = 'pt_bubble' # name of the model

smbhb = False # set to True if you want to overlay the new-physics signal to the SMBHB signal

parameters = {
    'log10_alpha':parameter.Uniform(-2,1)('log10_alpha'),
    'log10_T_star':parameter.Uniform(-4,4)('log10_T_star'), 
    'log10_H_R':parameter.Uniform(-3,0)('log10_H_R'), 
    'a':parameter.Uniform(1,3)('a'),
    'b':parameter.Uniform(1,3)('b'),
    'c':parameter.Uniform(1,3)('c')
    }

group = ['log10_alpha', 'log10_T_star', 'log10_H_R']


def S(x, a, b, c):
    """
    | Spectral shape as a functino of x=f/f_peak
    """
    return (a + b)**c / (b * x**(-a/c) + a * x**(b/c))**c
    


@aux.omega2cross
def spectrum(f, log10_alpha, log10_T_star, log10_H_R, a, b, c):
    """
    | Returns the GW energy density as a fraction of the 
    | closure density as a function of the parameters of the
    | model:
    |   - f
    |   - log10(alpha)
    |   - log10(T_star/Gev)
    |   - log10(H*R)
    |   - spectral shape parameters a,b,c
    """
    
    alpha = 10**log10_alpha
    T_star = 10**log10_T_star
    H_R = 10**log10_H_R

    H_beta = H_R * (8 * np.pi)**(-1/3)
    
    delta = 0.48 / (1 + 5.3 + 5) # velocity factor from 1605.01403
    f_peak = 0.35 / (1+ 0.69 +0.069) # peak frequency at emission (beta norm.) from 1605.01403
    p = 2 # alpha coefficient 
    q = 2 # rate coefficient 
    kappa = 1 # efficiency factor 

    g_s_eq = aux.g_s(aux.T_eq) # number of entropic relativistic dof at equality
    g_s_star = aux.g_s(T_star) # number of entropic relativistic dof at time of emission
    g_star = aux.g_rho(T_star) # number of relativistic dof at time of emission

    # normalization factor
    n = (a+b)/c
    norm = (
            (b/a)**(a/n)
            * (n * c / b)**c
            * gamma(a/n) * gamma(b/n)
            / (n * gamma(c))
    )

    # dilution factor 
    dil = (
            np.pi**2 / 90
            * g_star * (g_s_eq / g_s_star)**(4/3)
            * aux.T_0**4 / (aux.M_pl * aux.H_0)**2
            )
   

    # peak frequncy today in Hz
    f_0 = (
            90**(-1/2)
            * np.pi * g_star**(1/2) * (g_s_eq/g_s_star)**(1/3)
            * aux.T_0 / aux.M_pl
            * T_star * f_peak * H_beta**-1
            * nat.convert(nat.GeV, nat.Hz)
            )

    return (
            norm
            * aux.h**2 * dil
            * delta
            * (H_beta)**q
            * (kappa * alpha / (1 + alpha)) ** p
            * S(f / f_0, a, b, c)
            )

