import os
import numpy as np
import enterprise.signals.parameter as parameter
import src.models_utils as aux

name = 'PBH_log' # name of the model 

smbhb = True # set to True if you want to overlay the new-physics signal to the SMBHB signal

parameters = {
    'log10_fpeak' : parameter.Uniform(-11, -5)('log10_fpeak'),
    'log10_A' : parameter.Uniform(-3,1)('log10_A'),
    'log10_width' : parameter.Uniform(-0.31,0.39)('log10_width')
    
    }

group = []   

cwd = os.getcwd()
spectrum_file = aux.spec_importer(cwd +'/inputs/models/models_data/data.txt')

@aux.omega2cross
def spectrum(f,log10_A, log10_width,log10_fpeak):

    # f_peak = 10**log10_fpeak
    # f = 10**log10_f
    # A = 10**log10_A
    width = 10**log10_width


    
    return 10**spectrum_file(np.log10(f),log10_A = log10_A, width = width, log10_fpeak = log10_fpeak)
