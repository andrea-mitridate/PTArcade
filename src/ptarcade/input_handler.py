import os
import sys
import inspect
import optparse
import numpy as np
from enterprise_extensions import model_utils


class bcolors:
    HEADER = "\033[95m"
    OKBLUE = "\033[94m"
    OKCYAN = "\033[96m"
    OKGREEN = "\033[92m"
    WARNING = "\033[0;33m"
    FAIL = "\033[0;31m"
    ENDC = "\033[0m"
    BOLD = "\033[1m"
    UNDERLINE = "\033[4m"


def import_file(full_name, path):
    """
    Import a python module from a path. 3.4+ only.
    Does not call sys.modules[full_name] = path
    """
    from importlib import util

    spec = util.spec_from_file_location(full_name, path)
    mod = util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def get_cmdline_arguments():
    """
        Returns dictionary of command line arguments supplied to PhonoDark.
    """

    parser = optparse.OptionParser()
    parser.add_option('-m', action="store", default="",
            help="Path to models file. Sets the details of the new physics signal.")
    parser.add_option('-c', action="store", default="",
            help="Path to configuration file. Sets details of the monte carlo run.")
    parser.add_option('-n', action="store", default="0",
            help="Specifies the number of the chain.")  

    options_in, args = parser.parse_args()

    options = vars(options_in)

    cmd_input_okay = False
    if options['m'] != '' and options['n'] != '' and  options['c'] != '':
        cmd_input_okay = True

    return options, cmd_input_okay


def load_inputs(input_options):
    """
        Load the input parameters from the relevant files.
    """

    models_input = input_options['m']
    num_input = input_options['c']

    model_input_mod_name = os.path.splitext(os.path.basename(models_input))[0]
    model_mod = import_file(model_input_mod_name, os.path.abspath(models_input))

    num_input_mod_name = os.path.splitext(os.path.basename(num_input))[0]
    num_mod = import_file(num_input_mod_name, os.path.abspath(num_input))

    return {
            "model": model_mod,
            "config": num_mod
            }


def check_config(config):
    # checks that all the parameters are present in the config file
    must = ["pta_data", "N_samples"]

    optional = [
        "mod_sel",
        "out_dir",
        "resume",
        "scam_weight",
        "am_weight",
        "de_weight",
        "red_components",
        "corr",
        "gwb_components",
        "bhb_th_prior",
        "A_bhb_logmin",
        "A_bhb_logmax",
        "gamma_bhb",
    ]

    optional_default = {
        "mod_sel" : False,
        "out_dir" : './chains/',
        "resume" : False,
        "scam_weight" : 30,
        "am_weight" : 15,
        "de_weight" : 50,
        "red_components" : 30,
        "corr" : False,
        "gwb_components" : 14,
        "bhb_th_prior" : True,
        "A_bhb_logmin": None,
        "A_bhb_logmax" : None,
        "gamma_bhb" : None,
    }

    
    for par in must:
        if not hasattr(config, par):
            error = (f"{bcolors.FAIL}ERROR{bcolors.ENDC}: " +
                    f"{par} not found in the configuration file.")
            sys.exit(error)

    for par in optional:
        if not hasattr(config, par):
            setattr(config, par, optional_default[par])
            message = ( f"\t{par} not found in the configuration file, " + 
                        f"it will be set to {optional_default[par]}.\n")
            print(message)

    # checks PTA data 
    if isinstance(config.pta_data, str):
        if config.pta_data in ['NG15', 'NG12', 'IPTA2']:
            pass 
        else:
            error = (f"{bcolors.FAIL}ERROR{bcolors.ENDC}: " +
                    f"The pta dataset {config.pta_data} is not included in PTArcade. " +
                    "Please, choose between 'NG15', 'NG12', 'IPTA2' "+
                    "or load your own data.")
            sys.exit(error)
    elif isinstance(config.pta_data, dict):
        if list(config.pta_data.keys()) != ['psrs_data', 'noise_data', 'emp_dist']:
            error = (f"{bcolors.FAIL}ERROR{bcolors.ENDC}: " +
                 "The keys of the pta_data dictionary in the configuration file " +
                 "need to be ['psrs_data', 'noise_data', 'emp_dist'].")
            sys.exit(error)
        elif not os.path.exists(config.pta_data['psrs_data']):
            error = (f"{bcolors.FAIL}ERROR{bcolors.ENDC}: " +
                 "The path in pta_data['psrs_data'] does not exist.")
            sys.exit(error)
        else:
            pass
    else:
        error = (f"{bcolors.FAIL}ERROR{bcolors.ENDC}: " +
                 "The 'pta_data' variable in the configuration file needs to be " +
                 "eithera string between 'NG15', 'NG12', 'IPTA2', " + 
                 "or a dictionary pointing to a set of PTA data.")
        sys.exit(error)

    # checks booleans variables
    bools = {'mod_sel' : config.mod_sel,
             'resume' : config.resume,
             'corr' : config.corr,
             'bhb_th_prior' : config.bhb_th_prior}
    
    for key, value in bools.items():
        if not isinstance(value, bool):
            error = (f"{bcolors.FAIL}ERROR{bcolors.ENDC}: " +
                     f"The variable {key} in the configuration file must be a boolean.")
            sys.exit(error)
    
    # checks integers 
    integers = {'N_samples' : config.N_samples,
                'scam_weight' : config.scam_weight,
                'am_weight' : config.am_weight,
                'de_weight' :config.de_weight,
                'red_components' :config.red_components,
                'red_gwb_components': config.gwb_components}
    
    for key, value in integers.items():
        if not isinstance(value, int):
            error = (f"{bcolors.FAIL}ERROR{bcolors.ENDC}: The " +
                     f"variable {key} in the configuration file must be an integer.")
            sys.exit(error)

    # checks bhb params 
    bhb_pars = {'A_bhb_logmin' : config.A_bhb_logmin,
                'A_bhb_logmax' : config.A_bhb_logmax,
                'gamma_bhb' : config.gamma_bhb}
    
    for key, value in bhb_pars.items():
        if not isinstance(value, (float,int)) and value is not None:
            error = (f"{bcolors.FAIL}ERROR{bcolors.ENDC}:" +
                     f"The variable {key} in the configuration file must "+
                     "be a number (integer or float), or set to None.")
            sys.exit(error)

    if config.bhb_th_prior and (
        config.A_bhb_logmin or config.A_bhb_logmax or config.gamma_bhb):
        warning = (
            f"{bcolors.WARNING}WARNING{bcolors.ENDC}: "
            + "Since bhb_th_prior is set to True, any value of A_bhb_logmin, "
            + "A_bhb_logmax, or gamma_bhb will be ignored.\n")
        print(warning)

    return


def check_model(model, psrs, red_components, gwb_components):

    # checks that all the parameters are present in the config file 
    optional = ['name',
                'smbhb']
    
    optional_default = {'name' : 'np_model',
                        'smbhb' : False}
    
    if not (hasattr(model, 'parameters')):
        error = (f"{bcolors.FAIL}ERROR{bcolors.ENDC}: " +
                "the model file needs to contain a parameter" +
                "dictionary.")
        sys.exit(error)
    if not (hasattr(model, 'signal') or hasattr(model, 'spectrum')):
        error = (f"{bcolors.FAIL}ERROR{bcolors.ENDC}: " +
                "the model file needs to contain either a spectrum" +
                "function or a signal function.")
        sys.exit(error)
    for par in optional:
        if not hasattr(model, par):
            setattr(model, par, optional_default[par])
            message = ( f"\t{par} not found in the model file, " + 
                        f"it will be set to {optional_default[par]}.\n")
            print(message)

    # check priors 

    # check that parameters name match the variable of the spectrum function
    try:
        args = inspect.getfullargspec(model.spectrum)[0]
        args.remove('f')
        signal_type = 'spectrum'
    except AttributeError:
        args = inspect.getfullargspec(model.signal)[0]
        args.remove('toas')
        signal_type = 'signal'
        if 'pos' in args:
            args.remove('pos')
    if list(model.parameters.keys()) != args:
        error = (f"{bcolors.FAIL}ERROR{bcolors.ENDC}: " +
                "in the model file, the keys of the parameter dictionary need to " +
                f"match the parameters of the {signal_type} function.")
        sys.exit(error)

    # check spectrum/signal function
    x0 = {}

    for name, par in model.parameters.items():
        try:
            x0[name] = par.sample()
        except AttributeError:
            x0[name] = par.value

    if hasattr(model, 'spectrum'):
        N_f = max(red_components, gwb_components)
        T = model_utils.get_tspan(psrs)
        f_tab = np.linspace(1 / T, N_f / T, N_f)

        try:
            spectrum_tab = model.spectrum(f=f_tab, **x0)
        except AttributeError:
            error = (f"{bcolors.FAIL}ERROR{bcolors.ENDC}: " +
                "frequency components you selected and for random values of "+
                "I tried to evaluate the spectrum function on the array of " +
                "the parameter contained in the prior range but it failed. " +
                "Please, check that the spectrum function can take a numpy " +
                "list of frequencies as argument, and that it is well defined " +
                "within the entire prior volume.")
            sys.exit(error)


        if np.shape(spectrum_tab) != np.shape(f_tab):
            error = (f"{bcolors.FAIL}ERROR{bcolors.ENDC}: " +
                "the output of the spectrum function needs to have the same " +
                "dimensions of the frequency list passed as argument.")
            sys.exit(error)

    else:
        tmin = np.min([p.toas.min() for p in psrs])
        tmax = np.max([p.toas.max() for p in psrs])
        toas_tab = np.linspace(tmin, tmax, 10)
    
        try:
            signal_tab = model.signal(toas_tab, **x0)
        except AttributeError:
            error = (f"{bcolors.FAIL}ERROR{bcolors.ENDC}: " +
                "I tried to evaluate the signal function on an array of " +
                "toas withing the observing time and for a set of model "+
                "parameters contained in the prior range but it failed. " +
                "Please, check that the signal function can take a numpy " +
                "list of toas as argument, and that it is well defined " +
                "within the entire prior volume.")
            sys.exit(error)

        if np.shape(signal_tab) != np.shape(toas_tab):
            error = (f"{bcolors.FAIL}ERROR{bcolors.ENDC}: " +
                "the output of the signal function needs to have the same " +
                "dimensions of the toas list passed as argument.")
            sys.exit(error)

    return
