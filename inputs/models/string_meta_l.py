import os
import numpy as np
import enterprise.signals.parameter as parameter
import src.models_utils as aux

name = 'string_meta_l' # name of the model 

smbhb = True # set to True if you want to overlay the new-physics signal to the SMBHB signal

parameters = {
    'log10_mu' : parameter.Uniform(-14, -3)('log10_mu'),
    'log10_k' : parameter.Uniform(7.0, 9.5)('log10_k')
    }

group = ['log10_mu', 'log10_k']

cwd = os.getcwd()
log_spectrum = aux.spec_importer(cwd +'/inputs/models/models_data/meta_l.dat')

@aux.omega2cross
def spectrum(f, log10_mu, log10_k):
    return 10**log_spectrum(np.log10(f), log10_mu=log10_mu, log10_k=log10_k)