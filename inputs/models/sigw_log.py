#implementing SIGW - PBH model: Gaussian (log-normal peak) function (see the power spectrum and the GW density in Pi and Sasaki, 2005.12306v2)

import os
import numpy as np
import enterprise.signals.parameter as parameter
import src.models_utils as aux

name = 'sigw_log' # name of the model 

smbhb = True # set to True if you want to overlay the new-physics signal to the SMBHB signal

parameters = {
    'log10_fpeak' : parameter.Uniform(-10, -6)('log10_fpeak'),
    'log10_A' : parameter.Uniform(-3,1)('log10_A'),
    'width' : parameter.Uniform(0.1,3)('width')
    
    }

group = ['log10_fpeak','width','log10_A']   

cwd = os.getcwd()
spectrum_file = aux.spec_importer(cwd +'/inputs/models/models_data/sigw_log.txt')

@aux.omega2cross
def spectrum(f,log10_A, width,log10_fpeak):
    
    
    """Calculate GW energy density.

    Returns the GW energy density as a fraction of the closure density as a
    function of the parameters of the model:

    :param float f: Frequency in Hz.
    :param float log10_f_peak: Frequency at Gaussian peak in Hz in log10 space.
    :param float log10_A: Dimensionless scaling factor in log10 space.
    :param float width: width of the Gaussian curve (power spectrum).
    :returns: GW energy density as a fraction of the closure density.
    :rtype: float
    """

    A = 10**log10_A
    
    prefactor = ( # assumes f_reheating > f 
        (aux.omega_r) * (aux.g_rho(f, is_freq=True) / aux.g_rho_0) *
        (aux.g_s_0 / aux.g_s(f, is_freq=True))**(4/3)
        )
     
    
    return aux.h**2 * A**2 * prefactor * 10**spectrum_file(np.log10(f), width = width, log10_fpeak = log10_fpeak)
