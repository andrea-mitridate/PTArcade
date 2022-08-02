import enterprise.signals.parameter as parameter
import src.models_utils as aux
import numpy as np

name = 'pt_bubble' # name of the model

mod_sel = True # set to True if you want to compare the model to the SMBHB signal

smbhb = True # set to True if you want to overlay the new-physics signal to the SMBHB signal
corr = False # set to True if you want to include spatial correlations in the analysis 

parameters = {
    'log10_alpha':parameter.Uniform(-2,1)('log10_alpha'),
    'log10_T':parameter.Uniform(-4,4)('log10_T'), 
    'log10_H_R':parameter.Uniform(-3,0.5)('log10_H_R'),
    'a':parameter.Constant(3)('a'),
    'b':parameter.Constant(4)('b'),
    'c':parameter.Constant(7/2)('c')
    }

group = []

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
    |   - f/Hz
    |   - log10(alpha)
    |   - log10(T_star/Gev)
    |   - log10(H * R)
    |   - spectral shape parameters a,b,c
    """
    
    alpha = 10**log10_alpha
    T_star = 10**log10_T_star
    H_R = 10**log10_H_R
    
    ### mechanism dependent ###
    delta = 0.513
    f_peak = 0.35 / (1+ 0.69 +0.07)
    p = 2
    q = 1
    kappa = alpha / (0.73 + 0.083 * alpha**(1/2) + alpha)
    ###

    ups = 1 - (1 + 4 * (8 * np.pi)**1/3 * H_R * (1 + alpha)**1/2 / (3 * kappa * alpha)**1/2)**-(1/2)

    dilution = 7.69 * 10**-5 * aux.g_rho(T_star)/ aux.g_s(T_star)**(4/3)
    f_0 = 1.13 * 10**-7 * f_peak * T_star * aux.g_s(T_star)**(1/6) / H_R

    return dilution * ups * S(f/f_0, a, b, c) * delta * H_R**q * (kappa * alpha)**p / (1 + alpha)**p


