import os
import numpy as np
import enterprise.signals.parameter as parameter
import src.models_utils as aux

name = 'superstrings' # name of the model 

smbhb = True # set to True if you want to overlay the new-physics signal to the SMBHB signal

parameters = {
    'log10_Gmu' : parameter.Uniform(-14, -6)('log10_Gmu'),
    'log10_P': parameter.Uniform(-3, 0)('log10_P')
    }

group = ['log10_Gmu', 'log10_P']   

cwd = os.getcwd()
log_spectrum = aux.spec_importer(cwd +'/inputs/models/models_data/superstrings.h5')

@aux.omega2cross
def spectrum(f, log10_Gmu, log10_P):
    return 10**log_spectrum(np.log10(f), log10_Gmu=log10_Gmu, log10_P=log10_P)
