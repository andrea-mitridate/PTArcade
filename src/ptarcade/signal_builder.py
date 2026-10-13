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
    """Modified powerlaw function.

    Defines a modified  powerlaw function that takes as input an
    array containing the values of the amplitude and spectral index.

    Parameters
    ----------
    f : NDArray
        Frequency array.
    log10_Agamma : NDArray
        Two component NDArray of [Log10(amplitude), spectral index].
    components : int
        Number of components for each frequency.

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
    """Modified powerlaw function.

    Powerlaw function modified to work with ceffyl, for the SMBHB signal. ceffyl passes
    each parameter to the psd as a kwarg named after it, so the arguments carry the
    `gw_bhb_` names, which keep them distinct from model parameters such as `log10_A`.

    Parameters
    ----------
    f : NDArray
        Frequency array.
    Tspan: fload
        Observation time.
    gw_bhb_log10_A : float
        Log10(amplitude)
    gw_bhb_gamma : float
        Spectral index

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
    """Modified powerlaw function.

    Defines a modified powerlaw function that takes as input an
    array containing the values of the amplitude and spectral index.
    This version is compatible with ceffyl.

    Parameters
    ----------
    f : NDArray
        Frequency array.
    Tspan: fload
        Observation time.
    gw_bhb : NDArray
        Two component NDArray of [Log10(amplitude), spectral index].
    components : int
        Count by this number in `f`.

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


def combine_with_smbhb(psd: Callable[..., jax.Array], np_params: list[str]) -> Callable[..., jax.Array]:
    """Add the SMBHB powerlaw to a new-physics spectrum.

    [discovery][] reads the parameter names of a spectrum off its signature and then
    calls it positionally, so the returned function carries an explicit signature
    `(f, df, *np_params, *SMBHB_PARAMS)`. The SMBHB names have to live in that signature,
    since the shared GP gives no prefix that would keep them apart from the
    new-physics parameters. Summing the two spectra into a single
    prior — rather than handing `discovery` a list of priors — keeps the analytic
    inverse of the correlated GP prior matrix, which is worth roughly a factor of
    three per likelihood gradient.

    Parameters
    ----------
    psd : Callable[..., jax.Array]
        The new-physics cross-power spectral density, taking `(f, df, *np_params)`.
    np_params : list[str]
        Names of the new-physics parameters, in the order `psd` takes them.

    Returns
    -------
    Callable[..., jax.Array]
        The summed spectrum, with an inspectable signature.

    """
    n_np = len(np_params)

    def combined(f, df, *args):
        return psd(f, df, *args[:n_np]) + ds.powerlaw(f, df, *args[n_np:])

    combined.__signature__ = inspect.Signature(
        [
            inspect.Parameter(par, inspect.Parameter.POSITIONAL_OR_KEYWORD)
            for par in ("f", "df", *np_params, *SMBHB_PARAMS)
        ]
    )

    return combined


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
    """
    Reads in list of enterprise Pulsar instances and returns a PTA
    object instantiated with user-supplied options.

    Parameters
    ----------
    psrs : list[Pulsar]
        List of enterprise Pulsar instances.
    model : ModuleType
        Object containing the parameters of the exotic
        signal. Defaults to None.
    noisedict : dict, optional
        Dictionary of pulsar noise properties. Defaults to None]
    pta_dataset : str, optional
        PTADataset object containing the data for the PTA. Defaults to None.
    bhb_th_prior : bool
        If set to True the prior for the bhb signal will be
        derived by fitting a 2D Gaussian to the distribution of A and gamma
        in the holodeck library astro-02-gw. Defaults to False.
    gamma_bhb : float, optional
        Fixed common bhb process spectral index value. If set to
        None we vary the spectral index over the range [0, 7]. Defaults to None.
    A_bhb_logmin : float, optional
        specifies lower prior on the log amplitude of the bhb
        common process. If set to None, -18 is used. Defaults to Non
    A_bhb_logmax : float, optional
        specifies upper prior on the log amplitude of the bhb
        common process. If set to None, -14 is used if gamma_bhb = 13/3, -11 is
        used otherwise. Defaults to None.
    corr : bool
        if set to True HD correlations are assumed for GWBs. Defaults to False.
    red_components : int
        number of frequency components for the intrinsic
        red noise. Defaults to 30.
    gwb_components : int
        number of frequency components for the common processes. Defaults to 14

    Returns
    -------
    signal_base.PTA
        PTA object instantiated with user-supplied options.

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
                name="gw_bhb",
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


def ceffyl_builder(inputs):

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

def discovery_builder(
    psrs: list[ds.Pulsar],
    model: ModuleType,
    corr: bool = False,
    red_components: int = 30,
    gwb_components: int = 14,
) -> ds.ArrayLikelihood:

    with jax.default_device("cpu"):
        Tspan = ds.getspan(psrs)
        smbhb = getattr(model, "smbhb", False)

        # stochastic process
        if hasattr(model, "spectrum"):
            globalgp_psd = aux.omega2cross(model.spectrum, likelihood="discovery", model_name=model.name)
            globalgp_psd.__signature__ = inspect.signature(model.spectrum)

            # add the SMBHB signal to the new-physics spectrum, so that the two share a
            # single GP and the correlated prior keeps its analytic inverse
            if smbhb:
                np_params = [par for par in inspect.signature(model.spectrum).parameters if par != "f"]
                globalgp_psd = combine_with_smbhb(globalgp_psd, np_params)

            pslmodels = (
                ds.PulsarLikelihood(
                    [
                        psr.residuals,
                        ds.makenoise_measurement(psr, psr.noisedict),
                        ds.makegp_ecorr(psr, psr.noisedict),
                        ds.makegp_timing(psr, svd=True),
                    ],
                )
                for psr in psrs
            )

            uncommon = [par for par, val in model.parameters.items() if not getattr(val, "common", True)]

            rngp = ds.makecommongp_fourier(psrs, ds.powerlaw, red_components, T=Tspan, name="red_noise")
            if corr:
                # a common=False parameter is rejected by input_handler.check_model here
                hdgp = ds.makeglobalgp_fourier(psrs, globalgp_psd, ds.hd_orf, gwb_components, T=Tspan, name=model.name)
                return ds.ArrayLikelihood(pslmodels, commongp=rngp, globalgp=hdgp)

            # only the common parameters are shared across pulsars. The rest keep the
            # per-pulsar names that discovery_numpyro_model_builder samples them under
            common = [par for par in model.parameters if par not in uncommon]

            curngp = ds.makecommongp_fourier(
                psrs,
                globalgp_psd,
                gwb_components,
                T=Tspan,
                common=common + (list(SMBHB_PARAMS) if smbhb else []),
                name=model.name,
            )
            return ds.ArrayLikelihood(pslmodels, commongp=[rngp, curngp])

        # deterministic delays
        delay_func = model.signal
        delay_func.__signature__ = inspect.signature(model.signal)

        pslmodels = (
            ds.PulsarLikelihood(
                [
                    psr.residuals,
                    ds.makenoise_measurement(psr, psr.noisedict),
                    ds.makegp_ecorr(psr, psr.noisedict),
                    ds.makegp_timing(psr, svd=True),
                    ds.makedelay(
                        psr,
                        delay_func,
                        common=[
                            par
                            for par, val in model.parameters.items()
                            if getattr(val, "common", True) # return false if no "common" attribute
                        ],
                        name=model.name,
                    ),
                ]
            )
            for psr in psrs
        )

        rngp = ds.makecommongp_fourier(psrs, ds.powerlaw, red_components, T=Tspan, name="red_noise")

        if not smbhb:
            return ds.ArrayLikelihood(pslmodels, commongp=rngp)

        # a deterministic model has no spectrum to fold the SMBHB signal into, so it gets
        # its own GP. ds.powerlaw is used directly here: the "gw_bhb" GP name supplies the
        # prefix that `combine_with_smbhb` has to carry in the parameter names instead.
        if corr:
            bhbgp = ds.makeglobalgp_fourier(
                psrs, ds.powerlaw, ds.hd_orf, gwb_components, T=Tspan, name="gw_bhb",
            )
            return ds.ArrayLikelihood(pslmodels, commongp=rngp, globalgp=bhbgp)

        bhbgp = ds.makecommongp_fourier(
            psrs, ds.powerlaw, gwb_components, T=Tspan, common=list(SMBHB_PARAMS), name="gw_bhb",
        )
        return ds.ArrayLikelihood(pslmodels, commongp=[rngp, bhbgp])

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


def discovery_numpyro_model_builder(
    psrs: list[ds.Pulsar],
    inputs: dict[str, ModuleType],
    likelihood_obj: ds.ArrayLikelihood,
    pta_dataset: str | None = None,
    bhb_th_prior: bool = False,
    gamma_bhb: float | None = None,
    A_bhb_logmin: float | None = None,
    A_bhb_logmax: float | None = None,
) -> ds.ArrayLikelihood:
    ln_likelihood = likelihood_obj.logL
    params = ln_likelihood.params

    red_noise_gamma_params = []
    red_noise_amp_params = []

    for p in params:
        if "red_noise_log10_A" in p:
            red_noise_amp_params.append(p)
        elif "red_noise_gamma" in p:
            red_noise_gamma_params.append(p)

    smbhb = getattr(inputs["model"], "smbhb", False)

    # SMBHB priors, keyed by numpyro site name. The site names match the ones ent_builder
    # uses, so that the chains can be post-processed the same way
    bhb_dists = {}
    if smbhb:
        # the SMBHB parameters are named by `discovery_builder`, which prefixes them with
        # the model name for a correlated spectrum. Read them back off the likelihood
        # rather than rebuilding that naming rule here.
        bhb_amp_param = next(p for p in params if p.endswith(SMBHB_PARAMS[0]))
        bhb_gamma_param = next(p for p in params if p.endswith(SMBHB_PARAMS[1]))

        # theory motivated priors are only available for the datasets in `bhb_priors`.
        # pta_dataset is a dict when the user supplies their own data, hence the isinstance
        bhb_th_pars = bhb_priors.get(pta_dataset) if isinstance(pta_dataset, str) else None
        if bhb_th_prior and bhb_th_pars is None:
            log.warning(
                "Theory motivated priors for the SMBHB signal parameters are available only "
                f"for {' and '.join(bhb_priors)}. Reverting back to log uniform prior for A "
                "and uniform prior for gamma.\n",
            )

        if bhb_th_prior and bhb_th_pars is not None:
            mu, sigma = bhb_th_pars
            bhb_dists["gw_bhb"] = dist.MultivariateNormal(jnp.asarray(mu), covariance_matrix=jnp.asarray(sigma))
        else:
            bhb_dists[SMBHB_PARAMS[0]] = dist.Uniform(*bhb_log10_A_range(gamma_bhb, A_bhb_logmin, A_bhb_logmax))
            if gamma_bhb is None:
                bhb_dists[SMBHB_PARAMS[1]] = dist.Uniform(0, 7)

    model_name = inputs["model"].name
    model_params = inputs["model"].parameters
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
                    name = f"{psr.name}_{model_name}_{par}"
                    params[name] = _sample_numpyro_site(name, prior_spec)

        # SMBHB priors
        if "gw_bhb" in bhb_dists:
            gw_bhb = numpyro.sample("gw_bhb", bhb_dists["gw_bhb"])
            params[bhb_amp_param], params[bhb_gamma_param] = gw_bhb[0], gw_bhb[1]
        elif smbhb:
            params[bhb_amp_param] = numpyro.sample(SMBHB_PARAMS[0], bhb_dists[SMBHB_PARAMS[0]])
            params[bhb_gamma_param] = (
                numpyro.sample(SMBHB_PARAMS[1], bhb_dists[SMBHB_PARAMS[1]])
                if SMBHB_PARAMS[1] in bhb_dists
                else gamma_bhb
            )

        # Pulsar red noise priors
        for amp, gam in zip(red_noise_amp_params, red_noise_gamma_params, strict=True):
            params[amp] = numpyro.sample(amp, dist.Uniform(-20, -11))
            params[gam] = numpyro.sample(gam, dist.Uniform(0, 7))

        numpyro.factor("ll", ln_likelihood(params))

        if constraints:
            # keyed by the unprefixed model parameter names, as spectrum(freqs, **kwargs)
            # expects
            cosmo_params = {par: params[common_key(par)] for par in model_params}
            numpyro.factor(
                "cosmo",
                aux.cosmo_lnlikelihood_discovery(cosmo_params, inputs["model"].spectrum, constraints),
            )

    # ranges for the SMBHB parameters, written to priors.json by do_sample. The 2d
    # gaussian is one vector site, which do_sample splits into gw_bhb_0/gw_bhb_1 columns;
    # it has no hard bounds, and plot_utils drops the NaN ranges
    discovery_model.smbhb_priors = {}
    for site, prior in bhb_dists.items():
        if prior.event_shape:
            for idx in range(prior.event_shape[0]):
                discovery_model.smbhb_priors[f"{site}_{idx}"] = [np.nan, np.nan]
        else:
            discovery_model.smbhb_priors[site] = aux.to_prior_range(prior)

    return discovery_model
