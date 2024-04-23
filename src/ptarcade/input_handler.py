"""Methods to handle inputs."""
from __future__ import annotations

import inspect
import logging
import optparse
import os
import warnings
from dataclasses import dataclass
from importlib import util
from importlib.resources import files
from types import ModuleType
from typing import Any

import numpy as np
from enterprise.pulsar import Pulsar
from enterprise_extensions import model_utils

log = logging.getLogger("rich")

@dataclass(frozen=True)
class bcolors:
    """Class to hold ANSI escape sequences.

    Attributes
    ----------
    HEADER : str
    OKBLUE : str
    OKCYAN : str
    OKGREEN : str
    WARNING : str
    FAIL : str
    ENDC : str
    BOLD : str
    UNDERLINE : str

    """

    HEADER: str = "\033[95m"
    OKBLUE: str = "\033[94m"
    OKCYAN: str = "\033[96m"
    OKGREEN: str = "\033[92m"
    WARNING: str = "\033[0;33m"
    FAIL: str = "\033[0;31m"
    ENDC: str = "\033[0m"
    BOLD: str = "\033[1m"
    UNDERLINE : str = "\033[4m"


def import_file(full_name: str, path: str) -> ModuleType :
    """Import a python module from a path.

    Parameters
    ----------
    full_name : str
        Full name of the module.
    path : str
        Path to the module.

    Returns
    -------
    mod : ModuleType
        The loaded module.

    Notes
    -----
    Python 3.4+ only. Does not call `sys.modules[full_name] = path`

    """
    spec = util.spec_from_file_location(full_name, path)
    mod = util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def get_cmdline_arguments() -> tuple[dict[str, Any], bool]:
    """Create dictionary of command line arguments supplied.

    Returns
    -------
    options : dict
        Dictionary containing command line arguments
    comd_input_okay : bool
        Boolean that is True if command line arguments are all present

    """
    default_config = files('ptarcade.data').joinpath('default_config.py')

    parser = optparse.OptionParser()
    parser.add_option('-m', action="store", default="",
            help="Path to models file. Sets the details of the new physics signal.")
    parser.add_option('-c', action="store", default=default_config,
            help="Path to configuration file. Sets details of the monte carlo run.")
    parser.add_option('-n', action="store", default="0",
            help="Specifies the number of the chain.")

    options_in, args = parser.parse_args()

    options = vars(options_in)

    cmd_input_okay = False
    if options['m'] != '':
        cmd_input_okay = True

    return options, cmd_input_okay


def load_inputs(input_options: dict[str, Any]) -> dict[str, ModuleType]:
    """Load the input parameters from the relevant files.

    Parameters
    ----------
    input_options : dict[str, Any]
        Command line options. Output of [get_cmdline_arguments][ptarcade.input_handler.get_cmdline_arguments]

    Returns
    -------
    dict[str, ModuleType]
        Dictionary that maps to loaded inputs as Python modules

    """
    models_input = input_options['m']
    config_input = input_options['c']

    model_input_mod_name = os.path.splitext(os.path.basename(models_input))[0]
    model_mod = import_file(model_input_mod_name, os.path.abspath(models_input))

    config_input_mod_name = os.path.splitext(os.path.basename(config_input))[0]
    config_mod = import_file(config_input_mod_name, os.path.abspath(config_input))

    return {
            "model": model_mod,
            "config": config_mod
            }


def check_config(config: ModuleType) -> None:
    """Validate the config file.

    Parameters
    ----------
    config : ModuleType
        The configuration file loaded as a Python module

    Raises
    ------
    SystemExit
        * If `config.pta_data` is not string matching PTA dataset or a dictionary.
        * If boolean parameters from `bools` in `config` are not `bool`.
        * If integer parameters from `integers` in `config` are not `int`.
        * If BHB pars from `bhb_pars` are not `int`, `float`, or `None`.
        If the path in `config.pta_data['psrs_data']` doesn't exist.
        * If the keys of the `config.pta_data` dictionary do match the correct form.
        * If `config.pta_data` contains an illegal value.

    """
    # checks that all the parameters are present in the config file
    default = {
           "pta_data" : 'NG15',
           "N_samples" : int(2e6),
           "mode" : 'enterprise',
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

    for par in default.keys():
        if not hasattr(config, par):
            setattr(config, par, default[par])
            message = ( f"[green bold]{par}[/] [underline]not found[/] in the configuration file, " +
                        f"it [underline]will be set to[/] [green bold]{default[par]}[/].\n")
            log.info(message,extra={"markup": True, "highlighter": None})

    # checks PTA data
    if isinstance(config.pta_data, str):
        if config.pta_data in ["NG15", "NG12", "EPTA2_full", "EPTA2_new", "IPTA2"]:
            pass
        else:
            error = (
                f"The pta dataset [red]{config.pta_data}[/] is not included in PTArcade. "
                "Please, choose between 'NG15', 'NG12', 'EPTA2_full', 'EPTA2_new', 'IPTA2' "
                "or load your own data."
            )
            log.error(error, extra={"markup":True})
            raise SystemExit

    elif isinstance(config.pta_data, dict):
        if list(config.pta_data.keys()) != ["psrs_data", "noise_data", "emp_dist"]:
            error = (
                "The keys of the pta_data dictionary in the configuration file "
                f"need to be {['psrs_data', 'noise_data', 'emp_dist']}.\n"
                f"The keys I got were {list(config.pta_data.keys())}."
            )
            log.error(error)
            raise SystemExit

        elif not os.path.exists(config.pta_data["psrs_data"]):
            error = f"The path '[red]{config.pta_data['psrs_data']}[/]' specified in [green]pta_data['psrs_data'][/] does not exist."
            log.error(error, extra={"markup":True, "highlighter":False})
            raise SystemExit

        else:
            pass
    else:
        error = (
            "The 'pta_data' variable in the configuration file needs to be "
            "either a string between 'NG15', 'NG12', 'EPTA2_full', 'EPTA2_new', 'IPTA2', "
            "or a dictionary pointing to a set of PTA data.\n"
            f"You supplied pta_data={config.pta_data}"
        )
        log.error(error)
        raise SystemExit
            
    # checks mod
    if isinstance(config.pta_data, str):
        if config.mode in ["enterprise", "ceffyl"]:
            pass
        else:
            error = (
                f"'{config.mode}' is not a valid run mode for PTArcade.\n"
                "PTArcade can be run in two modes: 'enterprise' "
                "or 'ceffyl'. Please select one of these two for the 'mode' "
                "parameter in the configuration file."
            )
            log.error(error)
            raise SystemExit

    # checks booleans variables
    bools = {
        "mod_sel": config.mod_sel,
        "resume": config.resume,
        "corr": config.corr,
        "bhb_th_prior": config.bhb_th_prior,
    }

    for key, value in bools.items():
        if not isinstance(value, bool):
            error = (
                f"The variable '{key}' in the configuration file must be a boolean.\n"
                f"You supplied {key}={bools[key]}."
            )
            log.error(error)
            raise SystemExit

    # checks integers
    integers = {
        "N_samples": config.N_samples,
        "scam_weight": config.scam_weight,
        "am_weight": config.am_weight,
        "de_weight": config.de_weight,
        "red_components": config.red_components,
        "red_gwb_components": config.gwb_components,
    }

    for key, value in integers.items():
        if not isinstance(value, int):
            error = (
                f"variable '{key}' in the configuration file must be an integer.\n"
                f"You supplied {key}={integers[key]}, type is {type(integers[key])}."
            )
            log.error(error)
            raise SystemExit

    # checks bhb params
    bhb_pars = {"A_bhb_logmin": config.A_bhb_logmin, "A_bhb_logmax": config.A_bhb_logmax, "gamma_bhb": config.gamma_bhb}

    for key, value in bhb_pars.items():
        if not isinstance(value, (float, int)) and value is not None:
            error = (
                f"The variable '{key}' in the configuration file must "
                "be a number (integer or float), or set to None.\n"
                f"You supplied {key}={bhb_pars[key]}, type is {type(bhb_pars[key])}."
            )
            log.error(error)
            raise SystemExit

    if config.bhb_th_prior and (config.A_bhb_logmin or config.A_bhb_logmax or config.gamma_bhb):
        warning = (
            "Since bhb_th_prior is set to True, any value of A_bhb_logmin, "
            "A_bhb_logmax, or gamma_bhb will be ignored.\n"
        )
        log.warning(warning)


def check_model(model: ModuleType, psrs: list[Pulsar], red_components: int, gwb_components: int, mode: str) -> None:
    """Validate model file.

    Parameters
    ----------
    model : ModuleType
        Loaded user configuration as python module.
    psrs : list[Pulsar]
        List of [enterprise.pulsar.Pulsar][] objects.
    red_components : int
        Number of red components.
    gwb_components : int
        Number of GWB components.

    Raises
    ------
    SystemExit
        * If the shape of the output spectrum doesn't have the same shape as the input frequency list.
        * If the output signal doesn't have the same shape as the input frequency list.
        * If model file doesn't contain a parameter dictionary `parameters`.
        * If model file doesn't contain `spectrum` or `signal`.
        * If model `spectrum` evaluation fails.
        * If model `signal` evaluation fails.
        * If the keys of the model file's parameter dictionary do not match the keys of the signal or spectrum function.

    """
    # checks that all the parameters are present in the config file
    optional = ["name", "smbhb"]

    optional_default = {"name": "np_model", "smbhb": False}

    if not (hasattr(model, "parameters")):
        error = "The model file needs to contain a parameter dictionary."
        log.error(error)
        raise SystemExit

    if not (hasattr(model, "signal") or hasattr(model, "spectrum")):
        error = (
            "The model file needs to contain either a 'spectrum' "
            "function or a 'signal' function."
        )
        log.error(error)
        raise SystemExit

    for par in optional:
        if not hasattr(model, par):
            setattr(model, par, optional_default[par])
            message = f"[bold]{par}[/] not found in the model file, it will be set to [bold]{optional_default[par]}[/].\n"
            log.info(message, extra={"markup": True})

    # check priors

    # check that parameters name match the variable of the spectrum function
    try:
        args = inspect.getfullargspec(model.spectrum)[0]
        args.remove("f")
        signal_type = "spectrum"
    except AttributeError:
        args = inspect.getfullargspec(model.signal)[0]
        args.remove("toas")
        signal_type = "signal"
        if "pos" in args:
            args.remove("pos")
    if list(model.parameters.keys()) != args:
        error = (
            "In the model file, the keys of the 'parameter' dictionary need to "
            f"match the parameters of the {signal_type} function.\n"
            f"You supplied:\nmodel parameters = {list(model.parameters.keys())}\n"
            f"{signal_type} parameters = {args}"
        )
        log.error(error)
        raise SystemExit

    # check spectrum/signal function
    x0 = {}

    for name, par in model.parameters.items():
        try:
            x0[name] = par.sample()  # type: ignore
        except AttributeError:
            x0[name] = par.value  # type: ignore
        except TypeError:
            x0[name] = par(name).sample()

    if hasattr(model, "spectrum"):
        if mode == "enterprise":
            T = model_utils.get_tspan(psrs)
        else:
            T = 3 * 10**7
        N_f = max(red_components, gwb_components)
        f_tab = np.linspace(1 / T, N_f / T, N_f)

        try:
            spectrum_tab = model.spectrum(f=f_tab, **x0)
        except AttributeError:
            error = (
                "I tried to evaluate the spectrum function on the array of "
                "frequency components you selected and for random values of "
                "the parameter contained in the prior range but it failed. "
                "Please, check that the spectrum function can take a numpy "
                "list of frequencies as argument, and that it is well defined "
                "within the entire prior volume."
            )

            log.error(error)
            raise SystemExit from None

        if np.shape(spectrum_tab) != np.shape(f_tab):
            error = (
                "The output of the spectrum function needs to have the same "
                "dimensions of the frequency list passed as argument.\n"
                f"{np.shape(spectrum_tab)=} != {np.shape(f_tab)=}."
            )
            log.error(error)
            raise SystemExit

    elif hasattr(model, "signal") and mode == "ceffyl":
        error = ("You cannot use Ceffyl mode for deterministic signals.")
        log.error(error)
        raise SystemExit

    else:
        tmin = np.min([p.toas.min() for p in psrs])
        tmax = np.max([p.toas.max() for p in psrs])
        toas_tab = np.linspace(tmin, tmax, 10)

        try:
            signal_tab = model.signal(toas_tab, **x0)
        except AttributeError:
            error = (
                "I tried to evaluate the signal function on an array of "
                "toas withing the observing time and for a set of model "
                "parameters contained in the prior range but it failed. "
                "Please, check that the signal function can take a numpy "
                "list of toas as argument, and that it is well defined "
                "within the entire prior volume."
            )

            log.error(error)
            raise SystemExit from None

        if np.shape(signal_tab) != np.shape(toas_tab):
            error = (
                "The output of the signal function needs to have the same "
                "dimensions of the toas list passed as argument.\n"
                f"{np.shape(signal_tab)=} != {np.shape(toas_tab)=}."
            )

            log.error(error)
            raise SystemExit
        
    if hasattr(model, "orf"):
        if mode == "ceffyl":
            error = ("It is not possible to use user-specified ORF in ceffyl mode"
                 ", please use PTArcade in enterprise mode to do this.")
            log.error(error)
            raise SystemExit
        
        args = inspect.getfullargspec(model.orf)[0]
        if ['f', 'pos1', 'pos2'] != args[:3]:
            error = ("The first three arguments of the orf function should"
                 " be `f`, `pos1`, and `pos2` (even if the orf is not"
                 " frequency-dependent).")
            log.error(error)
            raise SystemExit
        args = [e for e in args if e not in ('f', 'pos1', 'pos2')]
        if list(model.parameters.keys()) != args:
            error = (
                "In addition to the parameters 'f', 'pos1', and 'pos2', the "
                "'orf' provided in the model file also needs to have as "
                "parameters all the ones contained in the 'parameter' dictionary."
                " (even if they are not used by the orf function)\n"
                f"The parameter dictionary contains {list(model.parameters.keys())}\n"
                "while the 'orf' function has the parameters ="
                f" {['f', 'pos1', 'pos2']+ args}"
            )
            log.error(error)
            raise SystemExit
