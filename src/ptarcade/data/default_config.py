pta_data = 'NG15' # PTA dataset to use in the analysis, available options are NG15, NG12, IPTA2

mode = 'ceffyl'

mod_sel = False # set to True if you want to compare the new-physics signal to the SMBHB signal

out_dir = './chains/'
resume = False
N_samples = int(2e6) # number of sample points for the mcmc

# intrinsic red noises parameters
red_components = 30 # number of frequency components for the intrinsic red noise

# bhbh signal parameters
corr = False # set to True if you want to include HD spatial correlations in the analysis 
gwb_components = 14 # number of frequency components for common process
bhb_th_prior = True # if set to True the prior for the bhb signal is set to a theory  motivated 2d gaussian
A_bhb_logmin = None # lower limit for the prior of the bhb signal amplitude. If set to None -18 is used
A_bhb_logmax = None # upper limit for the prior of the bhb signal amplitude. If set to None -14 is used
gamma_bhb = None # spectral index for the bhb singal. If set to None it's varied between [0, 7].
