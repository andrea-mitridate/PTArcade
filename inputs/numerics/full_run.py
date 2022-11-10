pta_data = 'NG15' # PTA dataset to use in the analysis, available options are NG15, NG12, IPTA2

# alterntively the pta_data can be specified creating a dictionary pta_data with the following structure
#pta_data = {
#'psrs_data': 'ng15_psrs_v1p1.pkl',
#'noise_data': None, # set to None if you do not want to use prederive noise parameters
#'emp_dist': None} # set to None if you do not want to use empirical distributions

mod_sel = True # set to True if you want to compare the new-physics signal to the SMBHB signal

# mcmc parameteres
out_dir = './chains/'
resume = True
N_samples = int(5e5) # number of sample points for the mcmc
scam_weight = 30
am_weight = 15
de_weight = 50

# intrinsic red noises parameters
red_components = 30 # number of frequency components for the intrinsic red noise

# bhbh signal parameters
corr = False # set to True if you want to include HD spatial correlations in the analysis 
gwb_components = 14 # number of frequency components for common process
A_bhb_logmin = None # lower limit for the prior of the bhb signal amplitude. If set to None -18 is used
A_bhb_logmax = None # upper limit for the prior of the bhb signal amplitude. If set to None -14 is used
gamma_bhb = 4.33 # spectral index for the bhb singal. If set to None it's varied between [0, 7].
