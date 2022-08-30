#implementing first PBH model: delta peak function (analytical solution for the spectrum)

import os
import numpy as np
import enterprise.signals.parameter as parameter
import src.models_utils as aux

name = 'PBH_delta' # name of the model 

smbhb = True # set to True if you want to overlay the new-physics signal to the SMBHB signal

parameters = {
    'log10_f_peak':parameter.Uniform(-11,-5)('log10_f_peak'),
    'log10_A':parameter.Uniform(-5,1)('log10_A')
    }

group = []   

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
    omega_r = 9.201e-5 # planck 2018 value
    #h = 0.674
    #g = 106.75
    #g0 = 3.38
    #gs = 106.75
    #gs0 = 3.94


    #def cg(g,g0,gs,gs0):

    #    return g/g0 * (gs0/gs)**(4/3)

    cg = 0.388
 
    f_s = f/f_peak

    common = (2 - 3 * f_s**2)

    factor1 = 3 / 64 * f_s**2 * (1 - f_s**2 / 4)**2
    factor2 = common**2 * np.heaviside(2-f_s, 0)
    factor3 = common**2 * np.pi**2 * np.heaviside(2 - np.sqrt(3) * f_s, 0)
    factor4 = common * np.log( np.abs( 1 - 4 / (3 * f_s**2) ) ) - 4

    return  aux.h**2 * cg  *  omega_r * A**2  * factor1 * factor2 * (factor3 + factor4**2)
