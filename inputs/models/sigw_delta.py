#implementing SIGW - PBH model: delta peak function (analytical solution for the spectrum, as shown in Yuan and Huang, 2103.04739v2)

import os
import numpy as np
import enterprise.signals.parameter as parameter
import src.models_utils as aux

name = 'sigw_delta' # name of the model 

smbhb = True # set to True if you want to overlay the new-physics signal to the SMBHB signal

parameters = {
    'log10_f_peak':parameter.Uniform(-11,-5)('log10_f_peak'),
    'log10_A':parameter.Uniform(-5,1)('log10_A')
    }

group = ['log10_f_peak','log10_A']   

@aux.omega2cross
def spectrum(f: float, log10_f_peak: float, log10_A: float) -> float:
    
    """Calculate GW energy density.

    Returns the GW energy density as a fraction of the closure density as a
    function of the parameters of the model:

    :param float f: Frequency in Hz.
    :param float log10_f_peak: Frequency at dirac delta peak in Hz in log10 space.
    :param float log10_A: Dimensionless scaling factor in log10 space.
    :returns: GW energy density as a fraction of the closure density.
    :rtype: float
    """
    
    f_peak = 10**log10_f_peak
    A = 10**log10_A
   
 
    f_s = f/f_peak

    common = (2 - 3 * f_s**2)

    factor1 = 3 / 64 * f_s**2 * (1 - f_s**2 / 4)**2
    factor2 = common**2 * np.heaviside(2-f_s, 0)
    factor3 = common**2 * np.pi**2 * np.heaviside(2 - np.sqrt(3) * f_s, 0)
    factor4 = common * np.log( np.abs( 1 - 4 / (3 * f_s**2) ) ) - 4
    
    prefactor = ( # assumes f_reheating > f 
        (aux.omega_r) * (aux.g_rho(f, is_freq=True) / aux.g_rho_0) *
        (aux.g_s_0 / aux.g_s(f, is_freq=True))**(4/3)
        )
  

    return  aux.h**2 * prefactor * A**2  * factor1 * factor2 * (factor3 + factor4**2)
