"""Module for building PTA signals."""
from __future__ import annotations

import inspect
import logging
import sys
from collections.abc import Callable
from pathlib import Path
from types import ModuleType
from zipfile import ZipFile

import discovery as ds
import jax
import jax.numpy as jnp
import numpy as np
import numpyro
from astropy.utils.data import download_file
from ceffyl import Ceffyl
from enterprise import constants as const
from enterprise.pulsar import Pulsar
from enterprise.signals import (deterministic_signals, gp_signals, parameter,
                                signal_base, utils)
from enterprise.signals.parameter import function
from enterprise_extensions import chromatic as chrom
from enterprise_extensions import hypermodel, model_utils
from enterprise_extensions.blocks import (common_red_noise_block,
                                          dm_noise_block, red_noise_block,
                                          white_noise_block)
from enterprise_extensions.sampler import get_parameter_groups
from numpy.typing import NDArray
from numpyro import distributions as dist

import ptarcade.ent_mod as mods
import ptarcade.models_utils as aux

log = logging.getLogger("rich")


# gaussian parameters for the SMBHB signal.
# NG15 parameters are extracted from the holodeck library astro-02-gw
# IPTA2 parameters are taken from Middleton et al. 2021
bhb_priors = {"NG15" : [np.array([-15.61492963, 4.70709637]), np.array([[0.27871359, -0.00263617], [-0.00263617, 0.12415383]])],
              "IPTA2" : [np.array([-15.02928454, 4.14290127]), np.array([[0.06869369, 0.00017051], [0.00017051, 0.04681747]])]}

# names carried by the SMBHB parameters in discovery and ceffyl modes. In discovery they
# are prefixed by the model name when the signal is correlated, so match them with
# str.endswith.
SMBHB_PARAMS = ("gw_bhb_log10_A", "gw_bhb_gamma")


def bhb_log10_A_range(
    gamma_bhb: float | None, A_bhb_logmin: float | None, A_bhb_logmax: float | None
) -> tuple[float, float]:
    """Bounds of the log-uniform SMBHB amplitude prior, shared by every backend.

    Parameters
    ----------
    gamma_bhb : float, optional
        Fixed SMBHB spectral index, or None if it is sampled.
    A_bhb_logmin : float, optional
        User lower bound. If None, -18 is used.
    A_bhb_logmax : float, optional
        User upper bound. If None, -14 is used if `gamma_bhb` = 13/3, -11 otherwise.

    Returns
    -------
    tuple[float, float]
        Lower and upper bounds on log10_A. Each side defaults independently.

    """
    logmin = -18 if A_bhb_logmin is None else A_bhb_logmin
    if A_bhb_logmax is not None:
        logmax = A_bhb_logmax
    elif gamma_bhb is not None and np.isclose(gamma_bhb, 13 / 3):
        logmax = -14
    else:
        logmax = -11
    return logmin, logmax


def unique_sampling_groups(super_model: hypermodel.Hypermodel) -> list[list[int]]:
    """Fix the hypermodel group structure.

    Parameters
    ----------
    super_model : enterprise_extensions.hypermodel.Hypermodel
        The configured hypermodel from [enterprise_extensions][]

    Returns
    -------
    unique_groups : list[list[int]]
        Nested list of lists with unique indices for parameters.

    """
    unique_groups = []
    for p in super_model.models.values():
        groups = get_parameter_groups(p)
        for group in groups:
            check_group = []
            for idx in group:
                param_name = p.param_names[idx]
                check_group.append(super_model.param_names.index(param_name))
            if check_group not in unique_groups:
                unique_groups.append(check_group)

    return unique_groups


@parameter.function
def powerlaw2(f: NDArray, log10_Agamma: NDArray, components: int = 2) -> NDArray:
    """Compute a powerlaw PSD from one array of amplitude and spectral index.

    Defines a modified powerlaw function that takes as input an
    array containing the values of the amplitude and spectral index.

    Parameters
    ----------
    f : NDArray
        Frequency array.
    log10_Agamma : NDArray
        Two component NDArray of [Log10(amplitude), spectral index].
    components : int, optional
        Number of components for each frequency (default is 2).

    Returns
    -------
    NDArray
        The modified powerlaw.

    """
    df = np.diff(np.concatenate((np.array([0]), f[::components])))
    return (
        (10 ** log10_Agamma[0]) ** 2
        / 12.0
        / np.pi**2
        * const.fyr ** (log10_Agamma[1] - 3)
        * f ** (-log10_Agamma[1])
        * np.repeat(df, components)
    )


@parameter.function
def powerlaw(f, Tspan, gw_bhb_log10_A, gw_bhb_gamma):
    """Compute the SMBHB powerlaw PSD for ceffyl.

    Powerlaw function modified to work with ceffyl, for the SMBHB signal. ceffyl passes
    each parameter to the psd as a kwarg named after it, so the arguments carry the
    `gw_bhb_` names, which keep them distinct from model parameters such as `log10_A`.

    Parameters
    ----------
    f : NDArray
        Frequency array.
    Tspan : float
        Observation time.
    gw_bhb_log10_A : float
        Log10(amplitude).
    gw_bhb_gamma : float
        Spectral index.

    Returns
    -------
    NDArray
        The modified powerlaw.

    """
    return (
        (10**gw_bhb_log10_A) ** 2
        / 12.0
        / np.pi**2
        * const.fyr ** (gw_bhb_gamma - 3)
        * f ** (-gw_bhb_gamma)
        / Tspan  # divide by Tspan here
    )


@parameter.function
def powerlaw2_ceffyl(f: NDArray, Tspan: float, gw_bhb: NDArray, components: int = 2) -> NDArray:
    """Compute a powerlaw PSD from one array of amplitude and spectral index, for ceffyl.

    Defines a modified powerlaw function that takes as input an
    array containing the values of the amplitude and spectral index.
    This version is compatible with ceffyl.

    Parameters
    ----------
    f : NDArray
        Frequency array.
    Tspan : float
        Observation time.
    gw_bhb : NDArray
        Two component NDArray of [Log10(amplitude), spectral index].
    components : int, optional
        Count by this number in `f` (default is 2).

    Returns
    -------
    NDArray
        The modified powerlaw.

    """
    return (
        (10 ** gw_bhb[0]) ** 2
        / 12.0
        / np.pi**2
        * const.fyr ** (gw_bhb[1] - 3)
        * f ** (-gw_bhb[1])
        / Tspan
    )


def tnequad_conv(noisedict: dict) -> bool:
    """Check equad defintion.

    Checks if the TempoNest definition of equad is used in the white noise dictionary.

    Parameters
    ----------
    noisedict : dict
        Dictionary containing noise terms.

    Returns
    -------
    tnequad : bool
        Whether TempoNest equad is used.

    Raises
    ------
    SystemExit
        If the equad convention is not consistent.

    """
    t2equad = False
    tnequad = False

    for x in noisedict:
        if "tnequad" in x:
            tnequad = True
        elif "t2equad" in x:
            t2equad = True

    if t2equad and tnequad:
        err = "ERROR: The convention for equad is not consistent across the PTA data."
        raise SystemExit(err)

    return tnequad


def ent_builder(
    psrs: list[Pulsar],
    model: ModuleType | None = None,
    noisedict: dict | None = None,
    pta_dataset: str | None = None,
    bhb_th_prior: bool = False,
    gamma_bhb: float | None = None,
    A_bhb_logmin: float | None = None,
    A_bhb_logmax: float | None = None,
    corr: bool = False,
    red_components: int = 30,
    gwb_components: int = 14,
) -> signal_base.PTA:
    """Build the enterprise PTA of a model from a list of pulsars.

    Parameters
    ----------
    psrs : list[Pulsar]
        List of enterprise Pulsar instances.
    model : ModuleType | None, optional
        The user's model file, with the parameters of the new-physics signal (default is
        None, the SMBHB-only model).
    noisedict : dict | None, optional
        Dictionary of pulsar noise properties. If None, the white noise is varied
        (default is None).
    pta_dataset : str | None, optional
        Name of the PTA data set, used to look up the theory-motivated SMBHB prior
        (default is None).
    bhb_th_prior : bool, optional
        If True, the prior for the SMBHB signal is a 2D Gaussian fit to the distribution
        of A and gamma in the holodeck library astro-02-gw. Only available for NG15 and
        IPTA2 (default is False).
    gamma_bhb : float | None, optional
        Fixed SMBHB spectral index. If None, it is varied over [0, 7] (default is None).
    A_bhb_logmin : float | None, optional
        Lower bound of the SMBHB log amplitude prior. If None, -18 is used (default is
        None).
    A_bhb_logmax : float | None, optional
        Upper bound of the SMBHB log amplitude prior. If None, -14 is used if
        `gamma_bhb` = 13/3, -11 otherwise (default is None).
    corr : bool, optional
        If True, HD correlations are assumed for the GWBs (default is False).
    red_components : int, optional
        Number of frequency components of the intrinsic red noise (default is 30).
    gwb_components : int, optional
        Number of frequency components of the common processes (default is 14).

    Returns
    -------
    signal_base.PTA
        PTA object instantiated with user-supplied options.

    Raises
    ------
    SystemExit
        If the DM variation or equad conventions are not consistent across the pulsars.

    """
    # timing model
    tm = gp_signals.MarginalizingTimingModel(use_svd=True)
    s = tm

    # find the maximum time span to set the GW frequency sampling
    Tspan = model_utils.get_tspan(psrs)

    # add pulsar intrinsic red noise
    s += red_noise_block(psd="powerlaw", prior="log-uniform", Tspan=Tspan, components=red_components)

    # add common red noise from SMBHB
    if model is None or model.smbhb:
        if corr:
            orf = "hd"
        else:
            orf = None

        if bhb_th_prior and (pta_dataset == "NG15" or pta_dataset == "IPTA2"):

            mu, sigma = bhb_priors[pta_dataset]

            if model is None:
                log10_Agamma_gw = parameter.Normal(mu=mu, sigma=sigma, size=2)("gw_bhb")
            elif model.smbhb:
                log10_Agamma_gw = parameter.Normal(mu=mu, sigma=sigma, size=2)("gw_bhb_np")
            powerlaw_gw = powerlaw2(log10_Agamma=log10_Agamma_gw)

            if orf == "hd":
                s += gp_signals.FourierBasisCommonGP(
                    spectrum=powerlaw_gw, orf=utils.hd_orf(), components=gwb_components, Tspan=Tspan, name="gw_bhb"
                )

            else:
                s += gp_signals.FourierBasisGP(
                    spectrum=powerlaw_gw, components=gwb_components, Tspan=Tspan, name="gw_bhb"
                )

        else:
            if bhb_th_prior:
                print(
                    "WARNING: Theory motivated priors for the SMBHB singal parameters are available only for NG15 and IPTA2. Reverting back to log uniform prior for A and uniform prior for gamma.\n"
                )

            # common_red_noise_block only uses logmin/logmax when both are given, so
            # resolve the defaults here
            A_bhb_logmin, A_bhb_logmax = bhb_log10_A_range(gamma_bhb, A_bhb_logmin, A_bhb_logmax)
            s += common_red_noise_block(
                psd="powerlaw",
                prior="log-uniform",
                Tspan=Tspan,
                components=gwb_components,
                gamma_val=gamma_bhb,
                orf=orf,
                # the new-physics model's own copy, as with the theory-motivated prior
                name="gw_bhb" if model is None else "gw_bhb_np",
                logmin=A_bhb_logmin,
                logmax=A_bhb_logmax,
            )

    # add DM variations
    dm_var = [hasattr(psr, "dmx") for psr in psrs]  # check if dmx parameters are present in pulsars objects

    if all(dm_var):
        pass
    elif not any(dm_var):
        s += dm_noise_block(gp_kernel="diag", psd="powerlaw", prior="log-uniform", components=30, gamma_val=None)
    else:
        sys.exit("ERROR: The convention for DM variation is not consistent across the PTA data.")

    # add new-physics signal
    if model:
        if hasattr(model, "signal"):
            signal = function(model.signal)
            signal = signal(**model.parameters)
            np_signal = deterministic_signals.Deterministic(signal, name=model.name)

            s += np_signal

        elif hasattr(model, "spectrum"):
            spectrum = aux.omega2cross(model.spectrum, likelihood="enterprise")
            cpl_np = spectrum(**model.parameters)

            if corr and hasattr(model, "orf"):
                orf = function(model.orf)
                orf = orf(**model.parameters)

                np_gwb = mods.FourierBasisCommonGP(
                    spectrum=cpl_np,
                    orf=orf,
                    components=gwb_components,
                    Tspan=Tspan,
                    name=model.name)
            elif corr:
                orf = utils.hd_orf()
                np_gwb = gp_signals.FourierBasisCommonGP(
                    spectrum=cpl_np,
                    orf=orf,
                    components=gwb_components,
                    Tspan=Tspan,
                    name=model.name)
            else:
                np_gwb = gp_signals.FourierBasisGP(
                    spectrum=cpl_np,
                    components=gwb_components,
                    Tspan=Tspan,
                    name=model.name)

            s += np_gwb

    # add white-noise, and act on psr objects
    models = []

    if noisedict is None:
        white_vary = True
        tnequad = False
    else:
        white_vary = False
        tnequad = tnequad_conv(noisedict)

    for p in psrs:
        if "NANOGrav" in p.flags["pta"]:
            s2 = s + white_noise_block(vary=white_vary, inc_ecorr=True, tnequad=tnequad, select="backend")
        else:
            s2 = s + white_noise_block(vary=white_vary, inc_ecorr=False, tnequad=tnequad, select="backend")

        # add DM dip for J1713
        if "1713" in p.name and not any(dm_var):
            s2 += chrom.dm_exponential_dip(tmin=54500, tmax=55000, idx=2, sign=False, name="dmexp_1")
            if p.toas.max() / const.day > 57850:
                s2 += chrom.dm_exponential_dip(tmin=57300, tmax=57850, idx=2, sign=False, name="dmexp_2")

        models.append(s2(p))

    # set up PTA
    pta = signal_base.PTA(models)

    # set white noise parameters
    if noisedict is not None:
        pta.set_default_params(noisedict)

    return pta


def ceffyl_builder(inputs: dict[str, ModuleType]) -> Ceffyl.ceffyl:
    """Build the ceffyl PTA of a model.

    The free spectrum of the PTA data set in the config is downloaded from zenodo and
    cached, unless the user supplies their own.

    Parameters
    ----------
    inputs : dict[str, ModuleType]
        The user's inputs, with the model file under "model" and the config file under
        "config".

    Returns
    -------
    Ceffyl.ceffyl
        The ceffyl PTA, with the model's signals added.

    """
    if inputs["config"].pta_data == "IPTA2":

        if inputs["config"].corr:
            warning = (
            "For the IPTA DR2 data set, ceffyl mode is only "
            "available without spatial correlations. But do not worry, for this data set, "
            "including or not spatial correlations should not make a significant difference."
            "The code will default to 'corr=False'.\n"
            )
            log.warning(warning)
        # download from zenodo
        ceffyldl = download_file(
            "https://zenodo.org/record/10495907/files/ipta2_30f_fs%7Bcp%7D_ceffyl.zip?download=1",
            cache=True,
            pkgname="ptarcade",
            )
        # make a Path object for ease of use
        ceffyldl = Path(ceffyldl)

        # Check if we've unzipped or not
        # If not, unzip and name the same as the original download so that astropy can find it again
        if ceffyldl.is_file():
            tempdir = (ceffyldl.parent / "temp")
            # extract
            with ZipFile(ceffyldl) as zf:
                zf.extractall(tempdir)
            # delete original zip
            ceffyldl.unlink()
            # rename unzipped dir to original zip name
            tempdir.rename(ceffyldl)
        # find ipta data inside dir
        datadir = (ceffyldl / "ipta2_30f_fs{cp}_ceffyl")


    elif inputs["config"].pta_data == "NG15":

        if inputs["config"].corr:
            # download from zenodo
            ceffyldl = download_file(
                "https://zenodo.org/records/21844115/files/ceffyl_data.zip?download=1",
                cache=True,
                pkgname="ptarcade",
                )
            # make a Path object for ease of use
            ceffyldl = Path(ceffyldl)

            # Check if we've unzipped or not
            # If not, unzip and name the same as the original download so that astropy can find it again
            if ceffyldl.is_file():
                tempdir = (ceffyldl.parent / "temp")
                # extract
                with ZipFile(ceffyldl) as zf:
                    zf.extractall(tempdir)
                # delete original zip
                ceffyldl.unlink()
                # rename unzipped dir to original zip name
                tempdir.rename(ceffyldl)
            # find pta data inside dir, remove trailing space
            datadir = ceffyldl / "ceffyl_data/30f_fs{hd}_ceffyl"
            padded = datadir.with_name(datadir.name + " ")
            if padded.exists():
                padded.rename(datadir)
        else:

            # download from zenodo
            ceffyldl = download_file(
                "https://zenodo.org/records/21844115/files/ceffyl_data.zip?download=1",
                cache=True,
                pkgname="ptarcade",
                )
            # make a Path object for ease of use
            ceffyldl = Path(ceffyldl)

            # Check if we've unzipped or not
            # If not, unzip and name the same as the original download so that astropy can find it again
            if ceffyldl.is_file():
                tempdir = (ceffyldl.parent / "temp")
                # extract
                with ZipFile(ceffyldl) as zf:
                    zf.extractall(tempdir)
                # delete original zip
                ceffyldl.unlink()
                # rename unzipped dir to original zip name
                tempdir.rename(ceffyldl)
            # find ipta data inside dir
            datadir = (ceffyldl / "ceffyl_data/30f_fs{cp}_ceffyl/")

    elif inputs["config"].pta_data == "NG12":
        if inputs["config"].corr:
            warning = (
            "For the NANOGrav 12.5-year data set, ceffyl mode is only "
            "available without spatial correlations. But do not worry, for this data set, "
            "including or not spatial correlations should not make a significant difference."
            "The code will default to 'corr=False'.\n"
            )
            log.warning(warning)
        # download from zenodo
        ceffyldl = download_file(
            "https://zenodo.org/record/10495907/files/ng12p5_30f_fs%7Bcp%7D_ceffyl.zip?download=1",
            cache=True,
            pkgname="ptarcade",
            )
        # make a Path object for ease of use
        ceffyldl = Path(ceffyldl)

        # Check if we've unzipped or not
        # If not, unzip and name the same as the original download so that astropy can find it again
        if ceffyldl.is_file():
            tempdir = (ceffyldl.parent / "temp")
            # extract
            with ZipFile(ceffyldl) as zf:
                zf.extractall(tempdir)
            # delete original zip
            ceffyldl.unlink()
            # rename unzipped dir to original zip name
            tempdir.rename(ceffyldl)
        # find ipta data inside dir
        datadir = (ceffyldl / "ng12p5_30f_fs{cp}_ceffyl")

    else:
        datadir = Path(inputs["config"].pta_data["psrs_data"])


    ceffyl_pta = Ceffyl.ceffyl(datadir)

    # Ceffyl.signal can't take a parameter.Constant (it has no `size`), so constants go
    # through const_params, which ceffyl passes to the psd as kwargs
    params = []
    const_params = {}
    for name, par in inputs["model"].parameters.items():
        if isinstance(par, parameter.ConstantParameter):
            const_params[name] = par.value
        else:
            params.append(par)

    model = []

    model.append(Ceffyl.signal(N_freqs=inputs["config"].gwb_components,
                          psd=aux.omega2cross(inputs["model"].spectrum, likelihood="ceffyl"),
                          params=params,
                          const_params=const_params,
                          name=''))


    if inputs["model"].smbhb:
        mu, sigma = bhb_priors.get(inputs["config"].pta_data, np.array([False, False]))
        bhb_const_params = {}

        if mu.all() and inputs["config"].bhb_th_prior:
            bhb_params = [parameter.Normal(mu=mu, sigma=sigma, size=2)("gw_bhb")]
            bhb_signal = powerlaw2_ceffyl

        else:
            A_bhb_logmin, A_bhb_logmax = bhb_log10_A_range(
                inputs["config"].gamma_bhb, inputs["config"].A_bhb_logmin, inputs["config"].A_bhb_logmax
            )
            log10_A_bhb = parameter.Uniform(A_bhb_logmin, A_bhb_logmax)(SMBHB_PARAMS[0])

            # Ceffyl.signal can't take a parameter.Constant (it has no `size`), so a fixed
            # gamma goes through const_params, which ceffyl passes to the psd as a kwarg
            if inputs["config"].gamma_bhb is not None:
                bhb_params = [log10_A_bhb]
                bhb_const_params = {SMBHB_PARAMS[1]: inputs["config"].gamma_bhb}
            else:
                bhb_params = [log10_A_bhb, parameter.Uniform(0, 7)(SMBHB_PARAMS[1])]
            bhb_signal = powerlaw


        model.append(Ceffyl.signal(N_freqs=inputs["config"].gwb_components,
                          psd=bhb_signal,
                          params=bhb_params,
                          const_params=bhb_const_params,
                          name=''))

    ceffyl_pta.add_signals(model)
    return ceffyl_pta

def with_signature(fn: Callable[..., jax.Array], names: list[str]) -> Callable[..., jax.Array]:
    """Give a positional PSD named arguments, which discovery reads off its signature.

    Parameters
    ----------
    fn : Callable[..., jax.Array]
        A PSD, called as `fn(f, df, *args)`.
    names : list[str]
        The names of `args`, in order.

    Returns
    -------
    Callable[..., jax.Array]
        `fn`, with the signature `(f, df, *names)`.

    """
    def named(f, df, *args):
        return fn(f, df, *args)

    named.__signature__ = inspect.Signature(
        [inspect.Parameter(par, inspect.Parameter.POSITIONAL_OR_KEYWORD) for par in ("f", "df", *names)]
    )
    return named


def discovery_builder(
    psrs: list[ds.Pulsar],
    model: ModuleType,
    corr: bool = False,
    red_components: int = 30,
    gwb_components: int = 14,
) -> ds.ArrayLikelihood:
    """Build the discovery likelihood of `model`.

    The array-wide spectrum (the new-physics spectrum, the SMBHB, or both) shares one GP
    with the pulsars' red noise when it is uncorrelated, as discovery's
    `makepowerlaw_crn` does: the two use the same Fourier basis, so a single GP halves the
    basis and roughly doubles the speed of the likelihood. A correlated spectrum gets its
    own global GP.
    """
    with jax.default_device("cpu"):
        Tspan = ds.getspan(psrs)
        smbhb = getattr(model, "smbhb", False)
        bhb = "gw_bhb"
        bhb_params = list(SMBHB_PARAMS)
        spectrum = hasattr(model, "spectrum")

        # deterministic delays
        delay = None
        if hasattr(model, "signal"):
            delay_func = model.signal
            delay_func.__signature__ = inspect.signature(model.signal)

            def delay(psr):
                return ds.makedelay(
                    psr,
                    delay_func,
                    common=[
                        par
                        for par, val in model.parameters.items()
                        if getattr(val, "common", True) # return false if no "common" attribute
                    ],
                    name=model.name,
                )

        pslmodels = [
            ds.PulsarLikelihood(
                [
                    psr.residuals,
                    ds.makenoise_measurement(psr, psr.noisedict),
                    ds.makegp_ecorr(psr, psr.noisedict),
                    ds.makegp_timing(psr, svd=True),
                    *([delay(psr)] if delay else []),
                ]
            )
            for psr in psrs
        ]

        # the array-wide spectrum: the new-physics spectrum, the SMBHB powerlaw, or their sum.
        # discovery reads its parameter names off the signature. The global GP of a
        # correlated spectrum prefixes them with its name; in the uncorrelated case they
        # share a GP with the red noise, so the new-physics names carry the model name to
        # keep them apart from the red noise's log10_A and gamma. One summed spectrum, rather
        # than a list of priors, keeps the analytic inverse of the correlated GP prior
        np_params = [par for par in inspect.signature(model.spectrum).parameters if par != "f"] if spectrum else []
        np_names = np_params if corr else [f"{model.name}_{par}" for par in np_params]
        names = [*np_names, *(bhb_params if smbhb else [])]
        if spectrum:
            np_psd = aux.omega2cross(model.spectrum, likelihood="discovery", model_name=model.name)

            def gw_psd(f, df, *args):
                phi = np_psd(f, df, *args[:len(np_names)])
                return phi + ds.powerlaw(f, df, *args[len(np_names):]) if smbhb else phi

            psd = with_signature(gw_psd, names)
        else:
            psd = with_signature(ds.powerlaw, bhb_params) if smbhb else None

        if corr:
            # a common=False parameter is rejected by input_handler.check_model here
            rngp = ds.makecommongp_fourier(psrs, ds.powerlaw, red_components, T=Tspan, name="red_noise")
            if psd is None:
                likelihood = ds.ArrayLikelihood(pslmodels, commongp=rngp)
            else:
                # without a new-physics spectrum, the GP name supplies the SMBHB prefix
                psd, name = (psd, model.name) if spectrum else (ds.powerlaw, bhb)
                hdgp = ds.makeglobalgp_fourier(psrs, psd, ds.hd_orf, gwb_components, T=Tspan, name=name)
                likelihood = ds.ArrayLikelihood(pslmodels, commongp=rngp, globalgp=hdgp)
        else:
            # a per-pulsar new-physics parameter becomes `{psr}_red_noise_{model}_{par}`
            uncommon = [
                f"{model.name}_{par}" for par, val in model.parameters.items() if not getattr(val, "common", True)
            ] if spectrum else []
            n_components = max(red_components, gwb_components) if psd is not None else red_components

            def red_noise_psd(f, df, log10_A, gamma, *args):
                phi = ds.powerlaw(f, df, log10_A, gamma)
                if red_components < n_components:
                    phi = jnp.where(jnp.arange(len(f)) < 2 * red_components, phi, 0.0)
                if psd is None:
                    return phi
                n_gw = 2 * gwb_components
                return phi.at[:n_gw].add(psd(f[:n_gw], df[:n_gw], *args))

            gp = ds.makecommongp_fourier(
                psrs,
                with_signature(red_noise_psd, ["log10_A", "gamma", *names]),
                n_components,
                T=Tspan,
                common=[par for par in names if par not in uncommon],
                name="red_noise",
            )
            likelihood = ds.ArrayLikelihood(pslmodels, commongp=gp)

        # build logL now and drop the pulsars' noise kernels (about 7 GB for NG15): without
        # a delay, logL keeps only small reduced arrays, but ArrayLikelihood would hold the
        # kernels through psls and vsm for the whole run
        likelihood.logL
        del pslmodels, likelihood.psls, likelihood.vsm
        return likelihood


def _sample_numpyro_site(site_name: str, prior_spec):
    """Register a single numpyro sample site from a translated prior."""
    if isinstance(prior_spec, aux.LinearExpSpec):
        lin = numpyro.sample(
            f"{site_name}__lin", dist.Uniform(10.0**prior_spec.pmin, 10.0**prior_spec.pmax),
        )
        return numpyro.deterministic(site_name, jnp.log10(lin))
    if isinstance(prior_spec, aux.ConstantSpec):
        return numpyro.deterministic(site_name, jnp.asarray(prior_spec.value))
    return numpyro.sample(site_name, prior_spec)


def smbhb_numpyro_priors(
    prefix: str,
    pta_dataset: str | dict | None,
    bhb_th_prior: bool,
    gamma_bhb: float | None,
    A_bhb_logmin: float | None,
    A_bhb_logmax: float | None,
) -> dict[str, dist.Distribution]:
    """numpyro sites and priors of the SMBHB parameters named `prefix`.

    The theory-motivated prior is one 2d site, `prefix`, which do_sample splits into the
    `{prefix}_0`/`{prefix}_1` columns that enterprise writes; the uniform prior is
    `{prefix}_log10_A` and, unless `gamma_bhb` fixes it, `{prefix}_gamma`.
    """
    # pta_dataset is a dict when the user supplies their own data, hence the isinstance
    bhb_th_pars = bhb_priors.get(pta_dataset) if isinstance(pta_dataset, str) else None
    if bhb_th_prior and bhb_th_pars is not None:
        mu, sigma = bhb_th_pars
        return {prefix: dist.MultivariateNormal(jnp.asarray(mu), covariance_matrix=jnp.asarray(sigma))}
    dists = {f"{prefix}_log10_A": dist.Uniform(*bhb_log10_A_range(gamma_bhb, A_bhb_logmin, A_bhb_logmax))}
    if gamma_bhb is None:
        dists[f"{prefix}_gamma"] = dist.Uniform(0, 7)
    return dists


def discovery_numpyro_model_builder(
    psrs: list[ds.Pulsar],
    inputs: dict[str, ModuleType],
    likelihood_obj: ds.ArrayLikelihood,
    pta_dataset: str | None = None,
    bhb_th_prior: bool = False,
    gamma_bhb: float | None = None,
    A_bhb_logmin: float | None = None,
    A_bhb_logmax: float | None = None,
) -> Callable[[], None]:
    """Build the numpyro model of a discovery run, with the likelihood `likelihood_obj`."""
    model = inputs["model"]
    ln_likelihood = likelihood_obj.logL
    params = ln_likelihood.params
    likelihood_params = set(params)

    red_noise_gamma_params = []
    red_noise_amp_params = []

    for p in params:
        if "red_noise_log10_A" in p:
            red_noise_amp_params.append(p)
        elif "red_noise_gamma" in p:
            red_noise_gamma_params.append(p)
    # the pulsars in the order of the red noise sites' entries
    red_noise_psrs = [p.removesuffix("_red_noise_log10_A") for p in red_noise_amp_params]
    if [p.removesuffix("_red_noise_gamma") for p in red_noise_gamma_params] != red_noise_psrs:
        msg = (
            "red noise amplitudes and spectral indices do not match pulsar by pulsar: "
            f"{red_noise_amp_params}, {red_noise_gamma_params}"
        )
        raise ValueError(msg)

    # theory motivated priors are only available for the datasets in `bhb_priors`
    if bhb_th_prior and not (isinstance(pta_dataset, str) and pta_dataset in bhb_priors):
        log.warning(
            "Theory motivated priors for the SMBHB signal parameters are available only "
            f"for {' and '.join(bhb_priors)}. Reverting back to log uniform prior for A "
            "and uniform prior for gamma.\n",
        )
    bhb_options = (pta_dataset, bhb_th_prior, gamma_bhb, A_bhb_logmin, A_bhb_logmax)

    # the SMBHB sites' priors, and the likelihood's names of (log10_A, gamma). discovery
    # prefixes those with the model name for a correlated spectrum, so they are looked up,
    # as in common_key
    smbhb = getattr(model, "smbhb", False)
    bhb_dists = smbhb_numpyro_priors("gw_bhb", *bhb_options) if smbhb else {}
    bhb_keys = [next(p for p in params if p.endswith(par)) for par in SMBHB_PARAMS] if smbhb else []

    model_name = model.name
    model_params = model.parameters
    constraints = inputs["config"].cosmo_constraints

    def common_key(par: str) -> str:
        """Name a common model parameter as the likelihood knows it.

        discovery prefixes it with the model name for a correlated spectrum, but a
        deterministic delay keeps the bare name whatever `corr` is, so the key is looked
        up rather than derived from `corr`.
        """
        prefixed = f"{model_name}_{par}"
        return prefixed if prefixed in params else par

    if constraints:
        # The cosmo constraints evaluate the spectrum once, globally, so a per-pulsar
        # parameter has no single value to feed it. Fail here, rather than with a
        # "missing argument" TypeError from inside the traced model.
        uncommon = [par for par, val in model_params.items() if not getattr(val, "common", True)]
        if uncommon:
            msg = (
                f"cosmo_constraints={constraints} cannot be used with the non-common model "
                f"parameter(s) {uncommon}: the BBN/LVK likelihoods evaluate the spectrum once "
                "for the whole array, so every model parameter must be common."
            )
            raise ValueError(msg)

    def discovery_model():
        # new physics priors
        params = {}
        for par, val in model_params.items():
            prior_spec = aux.to_numpyro_prior(val)
            if getattr(val, "common", True):
                params[common_key(par)] = _sample_numpyro_site(par, prior_spec)
            else:
                for psr in psrs:
                    # the site keeps one name; in the likelihood, a per-pulsar spectrum
                    # parameter sits in the red-noise GP (discovery_builder)
                    name = f"{psr.name}_{model_name}_{par}"
                    key = f"{psr.name}_red_noise_{model_name}_{par}"
                    params[key if key in likelihood_params else name] = _sample_numpyro_site(name, prior_spec)

        # SMBHB priors. The 2d gaussian is one site; a fixed gamma has none
        if smbhb:
            values = {site: numpyro.sample(site, prior) for site, prior in bhb_dists.items()}
            log10_A, gamma = values["gw_bhb"] if "gw_bhb" in values else (
                values["gw_bhb_log10_A"], values.get("gw_bhb_gamma", gamma_bhb)
            )
            params |= dict(zip(bhb_keys, (log10_A, gamma), strict=True))

        # Pulsar red noise priors: one vector site per parameter rather than one scalar site
        # per pulsar. NUTS updates each site separately, which on GPU is a small kernel per
        # site and step: on NG15 this makes a leapfrog step ~3x faster
        with numpyro.plate("psr", len(red_noise_psrs)):
            log10_A = numpyro.sample("red_noise_log10_A", dist.Uniform(-20, -11))
            gamma = numpyro.sample("red_noise_gamma", dist.Uniform(0, 7))
        params |= dict(zip(red_noise_amp_params, log10_A, strict=True)) | dict(
            zip(red_noise_gamma_params, gamma, strict=True)
        )

        numpyro.factor("ll", ln_likelihood(params))

        if constraints:
            # keyed by the unprefixed model parameter names, as spectrum(freqs, **kwargs)
            # expects
            cosmo_params = {par: params[common_key(par)] for par in model_params}
            numpyro.factor("cosmo", aux.cosmo_lnlikelihood_discovery(cosmo_params, model.spectrum, constraints))

    # ranges for the SMBHB parameters, written to priors.json by do_sample. The 2d
    # gaussian is one vector site, which do_sample splits into {site}_0/{site}_1 columns;
    # it has no hard bounds, and plot_utils drops the NaN ranges
    discovery_model.smbhb_priors = {}
    for site, prior in bhb_dists.items():
        if prior.event_shape:
            for idx in range(prior.event_shape[0]):
                discovery_model.smbhb_priors[f"{site}_{idx}"] = [np.nan, np.nan]
        else:
            discovery_model.smbhb_priors[site] = aux.to_prior_range(prior)

    # chain columns of the vector sites, which do_sample would otherwise number: the red
    # noise is written per pulsar, under enterprise's names
    discovery_model.site_columns = {
        "red_noise_log10_A": red_noise_amp_params,
        "red_noise_gamma": red_noise_gamma_params,
    }

    return discovery_model
