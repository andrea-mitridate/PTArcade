import os
import optparse

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
            help="Models info. Sets the white and red noise, the ULDM signal, and the ephemeris model.")
    parser.add_option('-n', action="store", default="",
            help="Numeric info. Sets details of the monte carlo run.")
    parser.add_option('-c', action="store", default="",
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
    cwd = os.getcwd()

    input_dir = cwd + '/inputs'
    mod_dir = input_dir + '/models/'
    num_dir = input_dir + '/numerics/'

    models_input = input_options['m']
    num_input = input_options['n']

    model_input_mod_name = os.path.splitext(os.path.basename(models_input))[0]
    model_mod = import_file(model_input_mod_name, os.path.join(mod_dir, models_input))

    num_input_mod_name = os.path.splitext(os.path.basename(num_input))[0]
    num_mod = import_file(num_input_mod_name, os.path.join(num_dir, num_input))

    return {
            "model": model_mod,
            "numerics": num_mod, 
            "mod_names": {
                'm': model_input_mod_name, 
                'n': num_input_mod_name
                }
            }