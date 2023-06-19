"""Module for sampling using [PTMCMCSampler][] and [enterprise_extensions][]."""
from __future__ import annotations

import os
import warnings

from astropy.utils.exceptions import AstropyDeprecationWarning

warnings.filterwarnings('ignore', category=AstropyDeprecationWarning)

import platform
import time
from types import ModuleType
from typing import Any
import shutil
import numpy as np

from enterprise.pulsar import Pulsar
from enterprise.signals.signal_base import PTA
from enterprise_extensions import hypermodel
from ptarcade.input_handler import bcolors
from numpy._typing import _ArrayLikeFloat_co as array_like
from numpy.typing import NDArray
from PTMCMCSampler.PTMCMCSampler import PTSampler
from ceffyl import Sampler

from ptarcade import input_handler, pta_importer, signal_builder
from ptarcade.models_utils import ParamDict


def cpu_model() -> str:
    """Get CPU info."""
    try:
        import cpuinfo
        return cpuinfo.get_cpu_info()["brand_raw"]
    except ModuleNotFoundError:
        return "unknown CPU (for better info install py-cpuinfo)"


def get_user_args() -> tuple[dict[str, ModuleType], dict[str, Any]] :
    """Get CLI arguments

    Returns
    -------
    inputs : dict[str, ModuleType]
        Dictionary of loaded user-supplied modules.
    input_options : dict[str, Any]
        Dictionary of user-supplied input options.

    Raises
    ------
    SystemExit
        If CLI input is missing required args.

    """
    # parse command line inputs
    input_options, cmd_input_okay = input_handler.get_cmdline_arguments()

    if not cmd_input_okay:

        error = (f"{bcolors.FAIL}ERROR{bcolors.ENDC}: Model file must be present\n"
        "\t- This is added with the -m input flags. Add -h (--help) flags for more help.\n")

        raise SystemExit(error)

    inputs = input_handler.load_inputs(input_options)
    input_handler.check_config(inputs['config'])

    if not hasattr(inputs["model"], "group"):
        pars_dic = inputs["model"].parameters
        group = [par for par in pars_dic.keys() if pars_dic[par].common]

        setattr(inputs["model"], "group", group)

    inputs["model"].parameters = ParamDict(inputs["model"].parameters)

    return inputs, input_options


def get_user_pta_data(inputs: dict[str, Any]) -> tuple[list[Pulsar], dict | None, array_like | None ]:
    """Import user-specified PTA data.

    Parameters
    ----------
    inputs : dict[str, Any]
        User supplied modules

    Returns
    -------
    psrs : list[Pulsar]
        List of Pulsar objects
    noise_params : dict | None
        Dictionary containing noise data
    emp_dist : array_like | None
        The empirical distribution to use for sampling

    """
    # import pta data
    psrs, noise_params, emp_dist = pta_importer.pta_data_importer(inputs['config'].pta_data)

    print(f"\tloaded {len(psrs)} pulsars\n")
    
    return psrs, noise_params, emp_dist


def initialize_pta(inputs: dict[str, Any], psrs: list[Pulsar] | None, noise_params : dict | None ) -> dict[int, PTA]:
    """Initialize the PTA with the user input

    Parameters
    ----------
    psrs : list[Pulsar]
        list of pulsar objects
    inputs : dict[str, Any]
        User specified modules
    noise_params : dict, optional
        User specified noise params

    Returns
    -------
    dict[int, PTA]
        Dictionary of [enterprise.signals.signal_base.PTA][] objects configured with user inputs

    """

    input_handler.check_model(
        model=inputs['model'],
        psrs=psrs,
        red_components=inputs['config'].red_components,
        gwb_components=inputs['config'].gwb_components,
        mode=inputs["config"].mode)


    if inputs["config"].mode == "enterprise":
        pta = {}

        pta[0] = signal_builder.ent_builder(
            psrs=psrs,
            model=inputs['model'],
            noisedict=noise_params,
            pta_dataset=inputs['config'].pta_data,
            bhb_th_prior=inputs['config'].bhb_th_prior,
            gamma_bhb=inputs['config'].gamma_bhb,
            A_bhb_logmin=inputs['config'].A_bhb_logmin,
            A_bhb_logmax=inputs['config'].A_bhb_logmax,
            corr=inputs['config'].corr,
            red_components=inputs["config"].red_components,
            gwb_components=inputs["config"].gwb_components)

        if inputs["config"].mod_sel:
            pta[1] = pta[0]

            pta[0] = signal_builder.ent_builder(
                psrs=psrs,
                model=None,
                noisedict=noise_params,
                pta_dataset=inputs['config'].pta_data,
                bhb_th_prior=inputs['config'].bhb_th_prior,
                gamma_bhb=inputs['config'].gamma_bhb,
                A_bhb_logmin=inputs['config'].A_bhb_logmin,
                A_bhb_logmax=inputs['config'].A_bhb_logmax,
                corr=inputs['config'].corr,
                red_components=inputs["config"].red_components,
                gwb_components=inputs["config"].gwb_components)
            
    elif inputs["config"].mode == "ceffyl":

        pta = signal_builder.ceffyl_builder(inputs)
        
    return pta


def setup_sampler(
        inputs: dict[str, ModuleType],
        input_options: dict[str, Any],
        pta: dict[int, PTA] | None,
        emp_dist: array_like | None,
) -> tuple[PTSampler, NDArray]:
    """Setup the PTMCMC sampler

    Parameters
    ----------
    inputs : dict[str, ModuleType]
        Dictionary of loaded user-supplied modules.
    input_options : dict[str, Any]
        Dictionary of user-supplied input options.
    pta : dict[int, PTA]
        Dictionary of [enterprise.signals.signal_base.PTA][] objects configured with user inputs
    emp_dist : array_like, optional
        The empirical distribution to use for sampling

    Returns
    -------
    sampler : PTMCMCSampler.PTSampler
        Configured [PTMCMCSampler.PTSampler][]
    x0 : NDArray
        Initial sample.

    """
    out_dir = os.path.join(
        inputs["config"].out_dir, inputs["model"].name, f'chain_{input_options["n"]}')
    
    if not inputs["config"].resume and os.path.exists(out_dir):
        shutil.rmtree(out_dir)

    if inputs['config'].mode == "enterprise":
        super_model = hypermodel.HyperModel(pta)

        groups = signal_builder.unique_sampling_groups(super_model)

        if inputs["model"].group:
            idx_params = [super_model.param_names.index(pp) for pp in inputs["model"].group]
            [groups.append(idx_params) for _ in range(5)] # type: ignore

        # add nmodel index to group structure
        groups.extend([[len(super_model.param_names)-1]])

        sampler = super_model.setup_sampler(
            resume=inputs["config"].resume,
            outdir=out_dir,
            sample_nmodel=inputs["config"].mod_sel,
            groups=groups,
            empirical_distr=emp_dist)

        x0 = super_model.initial_sample()

        super_model.get_lnlikelihood(x0) # Cache now to make timing more accurate

    elif inputs["config"].mode == "ceffyl":

        sampler = Sampler.setup_sampler(pta,
            outdir=out_dir,
            logL=pta.ln_likelihood,
            logp=pta.ln_prior)

        x0 = pta.initial_samples()

    return sampler, x0


def do_sample(inputs: dict[str, Any], sampler: PTSampler, x0: NDArray) -> None:
    """Run the configured sampler.

    Parameters
    ----------
    inputs : dict[str, Any]
        The user specified modules.
    sampler : PTMCMCSampler.PTSampler
        The configured [PTMCMCSampler.PTSampler][].

    x0 : NDArray
        The inital sample.

    """
    N_samples = inputs["config"].N_samples

    print(f'--- Starting to sample {N_samples} samples... ---\n')

    sampler.sample(
        x0,
        N_samples,
        SCAMweight=inputs['config'].scam_weight,
        AMweight=inputs['config'].am_weight,
        DEweight=inputs['config'].de_weight)

    print('--- Done sampling. ---')


def main():
    """Read user inputs, set up sampler and models, and run sampler."""
    print('\n--- Starting to run ---\n')
    print("\tNode", platform.node(), cpu_model(),"\n");

    start_cpu = time.process_time()
    start_real = time.perf_counter()

    inputs, input_options = get_user_args()

    psrs = None
    noise_params = None
    emp_dist = None

    if inputs["config"].mode == "enterprise":
        print('--- Loading Pulsars and noise data ... ---\n')

        # import pta data
        psrs, noise_params, emp_dist = get_user_pta_data(inputs)

        print('--- Done loading Pulsars and noise data. ---\n')


    print('--- Initializing PTA ... ---\n')

    pta = initialize_pta(inputs, psrs, noise_params)

    print('--- Done initializing PTA. ---\n\n')

    sampler, x0 = setup_sampler(inputs, input_options, pta, emp_dist)

    print("\tSetup times (including first sample) {:.2f} seconds real, {:.2f} seconds CPU\n".format(
        time.perf_counter()-start_real, time.process_time()-start_cpu));

    start_cpu = time.process_time()
    start_real = time.perf_counter()

    do_sample(inputs, sampler, x0)

    real_time = time.perf_counter()-start_real
    cpu_time = time.process_time()-start_cpu

    N_samples = inputs["config"].N_samples
    print("\tSampling times {:.2f} seconds real =  {:.4f} s/sample, {:.2f} seconds CPU =  {:.4f} s/sample\n".format(
        real_time, real_time/N_samples, cpu_time, cpu_time/N_samples))


if __name__ == "__main__":
    main()
