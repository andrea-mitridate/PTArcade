"""Module for sampling using [PTMCMCSampler][] and [enterprise_extensions][]."""
from __future__ import annotations

import os
import warnings

import numpyro

from astropy.utils.exceptions import AstropyDeprecationWarning

from ptarcade import pta_importer

warnings.filterwarnings('ignore', category=AstropyDeprecationWarning)

import json
import logging
import platform
import secrets
import shutil
import sys
import time
from collections.abc import Callable
from types import ModuleType
from typing import Any

# Fix missinng astropy.erfa in some of our older dependencies
import erfa

sys.modules["astropy.erfa"] = erfa

import types

from pathlib import Path

import jax
import numpy as np
import pandas as pd
import rich
from ceffyl import Sampler
from enterprise.pulsar import Pulsar
from enterprise.signals.signal_base import PTA
from enterprise_extensions import hypermodel
from numpy._typing import _ArrayLikeFloat_co as array_like
from numpy.typing import NDArray
from PTMCMCSampler.PTMCMCSampler import PTSampler
from rich.panel import Panel

from ptarcade import console, input_handler, pta_importer, signal_builder
from ptarcade.models_utils import ParamDict, cosmo_lnlikelihood, to_prior_range
from ptarcade.input_handler import bcolors

log = logging.getLogger("rich")
numpyro.enable_x64(use_x64=True)

# fewest NUTS warmup steps in discovery mode, see do_sample
MIN_WARMUP = 1000


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

        error = ("Model file must be present\n"
        "\t- This is added with the -[blue bold]m[/] input flags. Add -[blue bold]h[/] (--[blue bold]help[/]) flags for more help.\n")

        log.error(error, extra={"markup":True})
        raise SystemExit

    inputs = input_handler.load_inputs(input_options)
    input_handler.check_config(inputs['config'])

    if not hasattr(inputs["model"], "group"):
        pars_dic = inputs["model"].parameters
        group = [par for par in pars_dic if getattr(pars_dic[par], "common", True)]

        inputs["model"].group = group

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
    """Initialize the PTA with the user input.

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
        mode=inputs["config"].mode,
        cosmo_constraints=inputs["config"].cosmo_constraints,
        corr=inputs["config"].corr)


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

    elif inputs["config"].mode == "discovery":
        discovery_psrs = pta_importer.convert_enterprise_pulsars_to_discovery(
            psrs, inputs["config"].pta_data, noisedict=noise_params
        )
        pta = signal_builder.discovery_builder(
            psrs=discovery_psrs,
            model=inputs["model"],
            corr=inputs["config"].corr,
            red_components=inputs["config"].red_components,
            gwb_components=inputs["config"].gwb_components,
        )
        pta.psrs = discovery_psrs

    return pta


def ceffyl_prior_draw(pta: Any) -> Callable[[NDArray, int, float], tuple[NDArray, float]] | None:
    """Build a PTMCMC jump proposal that redraws one ceffyl parameter from its prior.

    ceffyl's own prior draws (`Sampler.setup_sampler(..., jump=True)`) redraw a single
    component of a vector parameter but evaluate its logpdf on that scalar, which
    enterprise broadcasts over every component. For the 2d gaussian SMBHB prior this
    gives a wrong Hastings ratio and a biased posterior. Here a vector parameter is
    redrawn whole, so the full logpdf is the right proposal density.

    Parameters with a custom (callable) prior are left out, as in the enterprise branch of
    `setup_sampler`: their `UserParameter` sampler is a fixed point (x0), which would
    break detailed balance.

    Parameters
    ----------
    pta : ceffyl.Ceffyl.ceffyl
        ceffyl PTA object. Only its common-process signals are used, which are the only
        ones [ptarcade.signal_builder.ceffyl_builder][] adds.

    Returns
    -------
    Callable[[NDArray, int, float], tuple[NDArray, float]] | None
        Jump proposal with the signature PTMCMC expects, or None if no parameter can be
        drawn from its prior.

    """
    # (indices into the sample array, parameter) for every parameter
    blocks = [
        (idx, par)
        for s in pta.signals
        if s.CP
        for idx, par in zip(s.pmap, s.params)
        if "UserParameter" not in str(par)
    ]
    if not blocks:
        return None

    def draw_from_prior(x: NDArray, iter: int, beta: float) -> tuple[NDArray, float]:
        q = x.copy()
        idx, par = blocks[np.random.randint(len(blocks))]
        idx = idx[0] if len(idx) == 1 else idx

        q[idx] = par.sample()

        # forward-backward jump probability
        return q, float(par.get_logpdf(x[idx]) - par.get_logpdf(q[idx]))

    return draw_from_prior


def run_dir(inputs: dict[str, ModuleType]) -> Path:
    """Output directory of a model: `out_dir/<model name>`, or `out_dir/<model name>_mod_sel` with `mod_sel`.

    Model selection runs get their own directory, so that they never overwrite the plain run
    of the same model file.
    """
    suffix = "_mod_sel" if inputs["config"].mod_sel else ""
    return Path(inputs["config"].out_dir) / f'{inputs["model"].name}{suffix}'


def chain_dir(inputs: dict[str, ModuleType], input_options: dict[str, Any]) -> Path:
    """Output directory of one chain: `run_dir/chain_<n>`, `n` from the `-n` option."""
    return run_dir(inputs) / f'chain_{input_options["n"]}'


def setup_sampler(
        inputs: dict[str, ModuleType],
        input_options: dict[str, Any],
        pta: dict[int, PTA] | None,
        emp_dist: array_like | None,
) -> tuple[PTSampler, NDArray]:
    """Set up the PTMCMC sampler.

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
    chain_path = str(chain_dir(inputs, input_options))

    if not inputs["config"].resume and os.path.exists(chain_path):
        shutil.rmtree(chain_path)

    if inputs['config'].mode == "enterprise":
        super_model = hypermodel.HyperModel(pta, log_weights=np.log([1., inputs["model"].mod_sel_wgt]))

        groups = signal_builder.unique_sampling_groups(super_model)

        if inputs["model"].group:
            # Build the list of parameter indices corresponding to the user-specified group.
            # Each entry in inputs["model"].group is a parameter name (or base name for
            # multidimensional parameters that are indexed as "{name}_0", "{name}_1", ...).
            idx_params = []
            for pp in inputs["model"].group:
                if pp in super_model.param_names:
                    # Shared parameter: appears exactly once in param_names
                    idx_params.append(super_model.param_names.index(pp))
                else:
                    # Multidimensional parameter: collect all indexed variants "{pp}_0", "{pp}_1", ...
                    i = 0
                    while f"{pp}_{i}" in super_model.param_names:
                        idx_params.append(super_model.param_names.index(f"{pp}_{i}"))
                        i += 1
            # Add this parameter group to the sampler's group list multiple times so
            # that it is proposed more frequently during sampling.
            [groups.append(idx_params) for _ in range(5)] # type: ignore

        # add nmodel index to group structure
        groups.extend([[len(super_model.param_names)-1]])

        if inputs["config"].cosmo_constraints:
            spectrum = inputs["model"].spectrum
            model_params = inputs["model"].parameters
            constraints = inputs["config"].cosmo_constraints

            def _cosmo_lnlikelihood(self, x):
                return cosmo_lnlikelihood(self, x, spectrum, model_params, constraints)

            super_model.get_lnlikelihood = types.MethodType(_cosmo_lnlikelihood, super_model)

        sampler = super_model.setup_sampler(
            resume=inputs["config"].resume,
            outdir=chain_path,
            sample_nmodel=inputs["config"].mod_sel,
            groups=groups,
            empirical_distr=emp_dist)

        # Remove prior-draw proposals for UserParameter parameters. Their sampler
        # is a fixed point (x0), which breaks detailed balance for that jump type.
        _user_param_names = set()
        for _p in super_model.params:
            if "UserParameter" in str(_p):
                _base = str(_p).split(":")[0]
                _user_param_names.update(
                    [f"{_base}_{i}" for i in range(_p.size)] if _p.size else [_base]
                )
        if _user_param_names:
            sampler.propCycle = [
                prop for prop in sampler.propCycle
                if not (hasattr(prop, "name_list") and
                        all(n in _user_param_names for n in prop.name_list))
            ]

        x0 = super_model.initial_sample()

        super_model.get_lnlikelihood(x0) # Cache now to make timing more accurate

    elif inputs["config"].mode == "ceffyl":

        if inputs["config"].cosmo_constraints:
            spectrum = inputs["model"].spectrum
            constraints = inputs["config"].cosmo_constraints
            model_params = inputs["model"].parameters

            def _cosmo_lnlikelihood(x):
                return cosmo_lnlikelihood(pta, x, spectrum, model_params, constraints)

            ln_likelihood = _cosmo_lnlikelihood

        else:
            ln_likelihood = pta.ln_likelihood

        sampler = Sampler.setup_sampler(pta,
            outdir=chain_path,
            logL=ln_likelihood,
            logp=pta.ln_prior,
            jump=False)

        # ceffyl's prior draws are disabled above because they are wrong for vector
        # parameters, see `ceffyl_prior_draw`. Without any, a chain that starts where the
        # likelihood is -inf (spectrum above the top of the KDE grid) can stay there.
        # Weight 15 matches ceffyl's draw_from_prior (5) + draw_from_gwb_priors (10)
        prior_draw = ceffyl_prior_draw(pta)
        if prior_draw is not None:
            sampler.addProposalToCycle(prior_draw, 15)

        x0 = pta.initial_samples()
    elif inputs["config"].mode == "discovery":
        numpyro_model = signal_builder.discovery_numpyro_model_builder(
            pta.psrs,
            inputs,
            pta,
            pta_dataset=inputs["config"].pta_data,
            bhb_th_prior=inputs["config"].bhb_th_prior,
            gamma_bhb=inputs["config"].gamma_bhb,
            A_bhb_logmin=inputs["config"].A_bhb_logmin,
            A_bhb_logmax=inputs["config"].A_bhb_logmax,
        )
        # the default init_to_uniform draws an unbounded site (the 2d gaussian SMBHB prior)
        # from [-2, 2], i.e. log10_A ~ 0, where the likelihood is NaN. Start from a prior
        # draw instead
        # the target acceptance sets the step size that warmup adapts to: numpyro's 0.8 by
        # default, higher for posteriors with divergent transitions (see do_sample)
        sampler = numpyro.infer.NUTS(
            numpyro_model,
            init_strategy=numpyro.infer.init_to_sample,
            target_accept_prob=inputs["config"].target_accept,
        )
        x0 = None

    return sampler, x0


def do_sample(inputs: dict[str, Any], sampler: PTSampler, x0: NDArray, chain_path: Path | None = None) -> None:
    """Run the configured sampler.

    Parameters
    ----------
    inputs : dict[str, Any]
        The user specified modules.
    sampler : PTMCMCSampler.PTSampler
        The configured [PTMCMCSampler.PTSampler][].

    x0 : NDArray
        The inital sample.
    chain_path : Path, optional
        Where discovery mode writes the chain (enterprise and ceffyl modes get it from
        `setup_sampler`). Defaults to the directory of chain 0, see `chain_dir`.

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

        if inputs["config"].mode in ["enterprise", "ceffyl"]:
            try:
                sampler.sample(
                    x0,
                    N_samples,
                    SCAMweight=inputs["config"].scam_weight,
                    AMweight=inputs["config"].am_weight,
                    DEweight=inputs["config"].de_weight,
                   **getattr(inputs["config"], "ptmcmc_sampler_kwargs", {}),
                )
            except RuntimeError as e:
                err = ("There was an error while sampling. If this error involves autocorrelation time,\n"
                      "a temporary fix is to increase the number of samples in the configuration file.\n"
                      "We are actively working to upgrade the autocorrelation routines in our sampler.\n\n")
                console.print("\n\n")
                log.exception(err)
                raise SystemExit from None
        elif inputs["config"].mode == "discovery":
            # a fresh seed for every run unless the config fixes one; it is printed and saved
            # next to the chain, so any run can be repeated
            seed = inputs["config"].seed if inputs["config"].seed is not None else secrets.randbits(32)
            console.print(f"Random seed {seed} (set seed = {seed} in the configuration file to repeat this run)\n")
            key = jax.random.key(seed)

            # NUTS adapts its step size and mass matrix in windows during warmup; fewer than
            # ~1000 warmup steps (Stan's default) leaves them short and the adapted step size
            # varies a lot from run to run
            num_warmup = max(N_samples // 4, MIN_WARMUP)
            mcmc = numpyro.infer.MCMC(
                sampler, num_chains=1, progress_bar=True, num_warmup=num_warmup, num_samples=N_samples
            )
            # -potential energy = the log posterior in the unconstrained space. Its value
            # differs from enterprise's by the Jacobians, but it is only used for the chain's
            # autocorrelation (chains_utils.bf_bootstrap)
            energy, stats = "potential_energy", ("diverging", "num_steps")
            mcmc.run(key, extra_fields=(energy, *stats))

            # a vector site cannot go into a single dataframe column, so split it into one
            # column per component: the red noise under the per-pulsar names the model
            # gives in site_columns, the 2d gaussian SMBHB prior as 'gw_bhb_0'/'gw_bhb_1',
            # the names chains_utils maps enterprise's 'gw_bhb_np_0'/'gw_bhb_np_1' to
            site_columns = getattr(sampler.model, "site_columns", {})
            samples = {}
            for site, draws in mcmc.get_samples().items():
                draws = np.asarray(draws)
                if draws.ndim == 1:
                    samples[site] = draws
                else:
                    components = draws.reshape(len(draws), -1).T
                    columns = site_columns.get(site, [f"{site}_{idx}" for idx in range(len(components))])
                    samples |= dict(zip(columns, components, strict=True))
            samples["log_posterior"] = -np.asarray(mcmc.get_extra_fields()[energy])
            samples_df = pd.DataFrame(samples)

            parameter_dict = inputs["model"].parameters
            prior_dict = {}
            for key, val in parameter_dict.items():
                if getattr(val, "common", True):
                    prior_dict[key] = to_prior_range(val)
                else:
                    # a non-common parameter is sampled once per pulsar, under
                    # '{psr}_{model}_{par}', so give each of those columns the range
                    suffix = f"_{inputs['model'].name}_{key}"
                    prior_range = to_prior_range(val)
                    for col in samples_df.columns:
                        if col.endswith(suffix):
                            prior_dict[col] = prior_range
            prior_dict.update(getattr(sampler.model, "smbhb_priors", {}))
            chain_path = Path(chain_path or chain_dir(inputs, {"n": 0}))
            chain_path.mkdir(exist_ok=True, parents=True)
            samples_df.to_feather(chain_path / "chain_1.feather")
            with (chain_path / "priors.json").open("w") as f:
                json.dump(prior_dict, f)
            (chain_path / "seed.txt").write_text(f"{seed}\n")

            # NUTS diagnostics per sample, kept out of the chain's parameter columns
            extra = mcmc.get_extra_fields()
            sampler_stats = pd.DataFrame({name: np.asarray(extra[name]) for name in stats})
            sampler_stats.to_feather(chain_path / "sampler_stats.feather")
            n_div = int(sampler_stats["diverging"].sum())
            console.print(f"{n_div} divergent transitions in {len(sampler_stats)} samples, "
                          f"{sampler_stats['num_steps'].mean():.0f} leapfrog steps per sample on average")
            if n_div:
                log.warning(
                    f"{n_div} divergent transitions: the chain may be biased; see "
                    f"{chain_path / 'sampler_stats.feather'}. A higher target_accept in the configuration "
                    f"file (it is {inputs['config'].target_accept}; try 0.9 or 0.95) gives smaller steps, "
                    "which usually removes them at the cost of more steps per sample"
                )

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

    if inputs["config"].mode in ["enterprise", "discovery"]:
        with console.status("Loading Pulsars and noise data...", spinner="bouncingBall"):

            # import pta data
            psrs, noise_params, emp_dist = get_user_pta_data(inputs)

            console.print(f"[bold green]Done loading [blue]{len(psrs)}[/] Pulsars and noise data :heavy_check_mark:\n")


    #with console.status("Initializing PTA...", spinner="bouncingBall"):
    pta = initialize_pta(inputs, psrs, noise_params)
    console.print("[bold green]Done initializing PTA :heavy_check_mark:\n")


    #with console.status("Initializing Sampler...", spinner="bouncingBall"):
    sampler, x0 = setup_sampler(inputs, input_options, pta, emp_dist)
    console.print("[bold green]Done initializing Sampler :heavy_check_mark:\n")

    console.print(f"Done with all initializtions.\nSetup times (including first sample) {time.perf_counter()-start_real:.2f} seconds real, {time.process_time()-start_cpu:.2f} seconds CPU\n")

    start_cpu = time.process_time()
    start_real = time.perf_counter()

    do_sample(inputs, sampler, x0, chain_dir(inputs, input_options))

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
