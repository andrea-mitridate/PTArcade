"""Module for building PTA signals."""
from __future__ import annotations

import logging
import sys
from importlib.resources import files
from pathlib import Path
from types import ModuleType
from zipfile import ZipFile

import numpy as np
from astropy.utils.data import download_file
from ceffyl import Ceffyl
from enterprise import constants as const
from enterprise.pulsar import Pulsar
from enterprise.signals import (deterministic_signals, gp_signals, parameter,
                                signal_base, utils, white_signals)
from enterprise.signals.parameter import function
from enterprise_extensions import chromatic as chrom
from enterprise_extensions import hypermodel, model_utils
from enterprise_extensions.blocks import (common_red_noise_block,
                                          dm_noise_block, red_noise_block,
                                          white_noise_block)
from enterprise_extensions.sampler import get_parameter_groups
from numpy.typing import NDArray

import ptarcade.models_utils as aux
import ptarcade.ent_mod as mods

log = logging.getLogger("rich")

# gaussian parameters for the SMBHB signal. 
# NG15 parameters are extracted from the holodeck library astro-02-gw
# IPTA2 parameters are taken from Middleton et al. 2021
bhb_priors = {"NG15" : [np.array([-15.61492963, 4.70709637]), np.array([[0.27871359, -0.00263617], [-0.00263617, 0.12415383]])],
              "IPTA2" : [np.array([-15.02928454, 4.14290127]), np.array([[0.06869369, 0.00017051], [0.00017051, 0.04681747]])]}


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
def powerlaw(f, Tspan, log10_A, gamma):
    
    """Modified powerlaw function.

    Powerlaw function modified to work with ceffyl.

    Parameters
    ----------
    f : NDArray
        Frequency array.
    Tspan: fload
        Observation time.
    log10_A : float
        Log10(amplitude)
    gamma : float
        Spectral index

    Returns
    -------
    NDArray
        The modified powerlaw.

    """
    
    return (
        (10**log10_A) ** 2 / 12.0 / np.pi**2 * const.fyr ** (gamma - 3) * f ** (-gamma) / Tspan  # divide by Tspan here
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
    white_vary: bool = True,
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
    white_vary: bool, optional
        If set to False, efac is set to a constant value of 1.0. If True, all
        white noise parameters are sampled. If a noisedict is supplied, this
        bool is set to False. Defaults to True. 
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

    # add common red noise
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

        elif bhb_th_prior and pta_dataset != "NG15" and pta_dataset != "IPTA2":
            print(
                "WARNING: Theory motivated priors for the SMBHB singal parameters are available only for NG15 and IPTA2. Reverting back to log uniform prior for A and uniform prior for gamma.\n"
            )
            s += common_red_noise_block(
                    psd='powerlaw',
                    prior='log-uniform',
                    Tspan=Tspan,
                    components=gwb_components,
                    orf=orf,
                    name = 'gw_bhb')

        else:
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
            spectrum = aux.omega2cross(model.spectrum)
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
        tnequad = False
    else:
        white_vary = False
        tnequad = tnequad_conv(noisedict)

    # constant white noise for simulations
    if noisedict is None and white_vary is False:
        efac = parameter.Constant(1.0)
        wn = white_signals.MeasurementNoise(efac=efac)
        s2 = s + wn
        for p in psrs:
            models.append(s2(p))

    else:
        for p in psrs:
            if "NANOGrav" in p.flags["pta"]:
                s2 = s + white_noise_block(vary=white_vary, inc_ecorr=True, tnequad=tnequad, select="backend")
                if "1713" in p.name and not any(dm_var):
                    s3 = s2 + chrom.dm_exponential_dip(tmin=54500, tmax=55000, idx=2, sign=False, name="dmexp_1")
                    if p.toas.max() / const.day > 57850:
                        s3 += chrom.dm_exponential_dip(tmin=57300, tmax=57850, idx=2, sign=False, name="dmexp_2")
                    models.append(s3(p))
                else:
                    models.append(s2(p))
            else:
                s4 = s + white_noise_block(vary=white_vary, inc_ecorr=False, tnequad=tnequad, select="backend")
                if "1713" in p.name and not any(dm_var):
                    s5 = s4 + chrom.dm_exponential_dip(tmin=54500, tmax=55000, idx=2, sign=False, name="dmexp_1")
                    if p.toas.max() / const.day > 57850:
                        s5 += chrom.dm_exponential_dip(tmin=57300, tmax=57850, idx=2, sign=False, name="dmexp_2")
                    models.append(s5(p))
                else:
                    models.append(s4(p))

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
                "https://zenodo.org/record/10495907/files/ng15_30f_fs%7Bhd%7D_ceffyl.zip?download=1",
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
            datadir = (ceffyldl / "ng15_30f_fs{hd}_ceffyl")
        else:

            # download from zenodo
            ceffyldl = download_file(
                "https://zenodo.org/record/10495907/files/ng15_30f_fs%7Bcp%7D_ceffyl.zip?download=1",
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
            datadir = (ceffyldl / "ng15_30f_fs{cp}_ceffyl")

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

    ceffyl_pta = Ceffyl.ceffyl(datadir)

    params = list(inputs["model"].parameters.values())

    model = []

    model.append(Ceffyl.signal(N_freqs=inputs["config"].gwb_components,
                          psd=aux.omega2cross(inputs["model"].spectrum, ceffyl=True),  
                          params=params,
                          name=''))
    
    
    if inputs["model"].smbhb:
        mu, sigma = bhb_priors.get(inputs["config"].pta_data, np.array([False, False]))

        if mu.all() and inputs["config"].bhb_th_prior:
            bhb_params = [parameter.Normal(mu=mu, sigma=sigma, size=2)("gw_bhb")]
            bhb_signal = powerlaw2_ceffyl

        else:
            if inputs["config"].A_bhb_logmin:
                A_bhb_logmin = inputs["config"].A_bhb_logmin
            else:
                A_bhb_logmin = -18

            if inputs["config"].A_bhb_logmax:
                A_bhb_logmax = inputs["config"].A_bhb_logmax
            else:
                A_bhb_logmax = -14

            log10_A_bhb = parameter.Uniform(A_bhb_logmin, A_bhb_logmax)("log10_A")

            if inputs["config"].gamma_bhb:
                gamma_bhb = parameter.Constant(inputs["config"].gamma_bhb)('gamma')
            else:
                gamma_bhb = parameter.Uniform(0, 7)("gamma")

            bhb_params = [log10_A_bhb, gamma_bhb]
            bhb_signal = powerlaw

            
        model.append(Ceffyl.signal(N_freqs=inputs["config"].gwb_components,
                          psd=bhb_signal,  
                          params=bhb_params,
                          name=''))


    return ceffyl_pta.add_signals(model)