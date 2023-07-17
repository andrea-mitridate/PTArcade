"""Module for sampling using [PTMCMCSampler][] and [enterprise_extensions][]."""
from __future__ import annotations

import os
import warnings

from astropy.utils.exceptions import AstropyDeprecationWarning

warnings.filterwarnings('ignore', category=AstropyDeprecationWarning)

import logging
import platform
import shutil
import sys
import time
from types import ModuleType
from typing import Any

# Fix missinng astropy.erfa in some of our older dependencies
import erfa

sys.modules["astropy.erfa"] = erfa

import numpy as np
import rich
from ceffyl import Sampler
from enterprise.pulsar import Pulsar
from enterprise.signals.signal_base import PTA
from enterprise_extensions import hypermodel
from numpy._typing import _ArrayLikeFloat_co as array_like
from numpy.typing import NDArray
from PTMCMCSampler.PTMCMCSampler import PTSampler
from rich import print
from rich.console import Console
from rich.panel import Panel

from ptarcade import input_handler, pta_importer, signal_builder
from ptarcade.input_handler import bcolors
from ptarcade.models_utils import ParamDict
from ptarcade import console

log = logging.getLogger("rich")

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

        error = (f"Model file must be present\n"
        "\t- This is added with the -[blue bold]m[/] input flags. Add -[blue bold]h[/] (--[blue bold]help[/]) flags for more help.\n")

        log.error(error, extra={"markup":True})
        raise SystemExit

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
            white_vary=inputs['config'].white_vary,
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

    console.print(f"[bold green]Starting to sample {N_samples} samples\n")

    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            category=RuntimeWarning,
            message="invalid value encountered in scalar subtract",
            module="PTMCMCSampler",
            lineno=567,
        )

        warnings.filterwarnings(
            "ignore",
            category=RuntimeWarning,
            message="All-NaN axis encountered",
            module="PTMCMCSampler",
            lineno=464,
        )

        warnings.filterwarnings(
            "ignore",
            category=RuntimeWarning,
            message="divide by zero encountered in log",
            module="enterprise.signals.parameter",
            lineno=62,
        )
        try:
            sampler.sample(
                x0,
                N_samples,
                SCAMweight=inputs["config"].scam_weight,
                AMweight=inputs["config"].am_weight,
                DEweight=inputs["config"].de_weight,
            )
        except RuntimeError as e:
            err = ("There was an error while sampling. If this error involves autocorrelation time,\n"
                  "a temporary fix is to increase the number of samples in the configuration file.\n"
                  "We are actively working to upgrade the autocorrelation routines in our sampler.\n\n")
            console.print("\n\n")
            log.exception(err)
            raise SystemExit from None

    console.print()
    console.print(Panel.fit("[bold green]Done sampling[/]", border_style="green"))
    console.print()


def main():
    """Read user inputs, set up sampler and models, and run sampler."""
    console.print(Panel.fit('[bold green]Starting to run[/]', border_style="green"))
    console.print()
    table = rich.table.Table(title="Node Information", title_justify="left",box=rich.box.ROUNDED)

    table.add_column("Node", style="cyan", no_wrap=True)
    table.add_column("CPU", style="magenta")

    table.add_row(platform.node(), cpu_model())

    console.print(table)
    console.print()

    start_cpu = time.process_time()
    start_real = time.perf_counter()

    inputs, input_options = get_user_args()

    psrs = None
    noise_params = None
    emp_dist = None

    if inputs["config"].mode == "enterprise":
        with console.status("Loading Pulsars and noise data...", spinner="bouncingBall"):

            # import pta data
            psrs, noise_params, emp_dist = get_user_pta_data(inputs)

            console.print(f"[bold green]Done loading [blue]{len(psrs)}[/] Pulsars and noise data :heavy_check_mark:\n")


    with console.status("Initializing PTA...", spinner="bouncingBall"):
        pta = initialize_pta(inputs, psrs, noise_params)
        console.print("[bold green]Done initializing PTA :heavy_check_mark:\n")


    with console.status("Initializing Sampler...", spinner="bouncingBall"):
        sampler, x0 = setup_sampler(inputs, input_options, pta, emp_dist)
        console.print("[bold green]Done initializing Sampler :heavy_check_mark:\n")

    console.print("Done with all initializtions.\nSetup times (including first sample) {:.2f} seconds real, {:.2f} seconds CPU\n".format(
        time.perf_counter()-start_real, time.process_time()-start_cpu));

    start_cpu = time.process_time()
    start_real = time.perf_counter()

    do_sample(inputs, sampler, x0)

    real_time = time.perf_counter()-start_real
    cpu_time = time.process_time()-start_cpu

    N_samples = inputs["config"].N_samples

    summary_table = rich.table.Table(title="Run Summary", title_justify="left", box=rich.box.ROUNDED)

    summary_table.add_column("Time (real)", style="cyan")
    summary_table.add_column("Time (real)/sample", style="cyan")
    summary_table.add_column("Time (CPU)", style="magenta")
    summary_table.add_column("Time (CPU)/sample", style="magenta")


    summary_table.add_row(f"{real_time:.2f}", f"{real_time/N_samples:.4f}", f"{cpu_time:.2f}", f"{cpu_time/N_samples:.4f}")
    console.print(summary_table)


if __name__ == "__main__":
    main()
