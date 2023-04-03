import sys
import numpy as np

from enterprise.signals import utils
from enterprise.signals import signal_base
from enterprise.signals import parameter
from enterprise.signals import gp_signals
from enterprise.signals import deterministic_signals
from enterprise_extensions import model_utils
from enterprise import constants as const
from enterprise_extensions import chromatic as chrom
from enterprise_extensions.sampler import get_parameter_groups

from enterprise_extensions.blocks import (
                                          common_red_noise_block,
                                          dm_noise_block, red_noise_block,
                                          white_noise_block)


def unique_sampling_groups(super_model):
    """
    Fixes the hypermodel group structure
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
def powerlaw2(f, log10_Agamma, components=2):
    """
    Defines a modified  powerlaw function that takes as input an
    array containing the values of the amplitude and spectral index.
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


def tnequad_conv(noisedict):
    """
    Checks if the TempoNest definition of equad is used in 
    the white noise dictionary. 
    """
    t2equad = False
    tnequad = False

    for x in noisedict:
        if 'tnequad' in x:
            tnequad = True
        elif 't2equad' in x:
            t2equad = True
        
    if t2equad and tnequad:
        sys.exit('ERROR: The convention for equad is not consistent across the PTA data.')
    
    return tnequad
    

def builder(
    psrs, 
    model=None, 
    noisedict=None, 
    pta_dataset=None,
    bhb_th_prior=False,
    gamma_bhb=None, 
    A_bhb_logmin=None, 
    A_bhb_logmax=None,
    corr=False, 
    red_components=30, 
    gwb_components=14):
    """
    Reads in list of enterprise Pulsar instances and returns a PTA
    object instantiated with user-supplied options.
    :param model: object containing the parameters of the exotic
        signal. 
        [default = None]
    :param noisedict: Dictionary of pulsar noise properties.
        [default = None]
    :param bhb_th_prior: if set to True the prior for the bhb signal will be 
        derived by fitting a 2D Gaussian to the distribution of A and gamma 
        found in 2011.01246
        [default = False]
    :param gamma_bhb: fixed common bhb process spectral index value. If set to
        None we vary the spectral index over the range [0, 7].
        [default = None]
    :param A_bhb_logmin: specifies lower prior on the log amplitude of the bhb
        common process. If set to None, -18 is used.
        [default = None] 
    :param A_bhb_logmax: specifies upper prior on the log amplitude of the bhb
        common process. If set to None, -14 is used if gamma_bhb = 13/3, -11 is
        used otherwise.
        [default = None]
    :param corr: if set to True HD correlations are assumed for GWBs
        [default = False]
    :red_components: number of frequency components for the intrinsic
        red noise. 
        [default = 30]
    :gwb_components: number of frequency components for the common processes.
        [default = 14]
    """

    # timing model
    tm = gp_signals.MarginalizingTimingModel(use_svd=True)
    s = tm

    # find the maximum time span to set the GW frequency sampling 
    Tspan = model_utils.get_tspan(psrs)

    # add pulsar intrinsic red noise 
    s += red_noise_block(
        psd='powerlaw', 
        prior='log-uniform', 
        Tspan=Tspan,
        components=red_components)


    # add common red noise
    if model is None or model.smbhb:
        if corr:
            orf = 'hd'
        else:
            orf = None

        if bhb_th_prior and (pta_dataset=='NG15' or pta_dataset=='IPTA2'):
            # gaussian parameters extracted from 2011.01246
            if pta_dataset == 'NG15':
                mu = np.array([-15.54398667, 4.53454624])
                sigma = np.array([[0.29921382, -0.02242464], [-0.02242464, 0.10293028]])
            elif pta_dataset == 'IPTA2':
                mu = np.array([-15.02928454, 4.14290127])
                sigma = np.array([[0.06869369, 0.00017051], [0.00017051, 0.04681747]])
            
            if model is None:
                log10_Agamma_gw = parameter.Normal(mu=mu, sigma=sigma , size=2)('gw_bhb')
            elif model.smbhb:
                log10_Agamma_gw = parameter.Normal(mu=mu, sigma=sigma , size=2)('gw_bhb_np')
            powerlaw_gw = powerlaw2(log10_Agamma=log10_Agamma_gw)

            if orf == 'hd':
                s += gp_signals.FourierBasisCommonGP(
                    spectrum=powerlaw_gw,
                    orf=utils.hd_orf(),
                    components=gwb_components,
                    Tspan=Tspan,
                    name='gw_bhb')

            else:
                s += gp_signals.FourierBasisGP(
                    spectrum=powerlaw_gw,
                    components=gwb_components,
                    Tspan=Tspan,
                    name='gw_bhb')

        elif bhb_th_prior and pta_dataset!='NG15' and pta_dataset!='IPTA2':
            print('WARNING: Theory motivated priors for the SMBHB singal parameters are available only for NG15 and IPTA2. Reverting back to log uniform prior for A and uniform prior for gamma.\n')
            s += common_red_noise_block(
                    psd='powerlaw', 
                    prior='log-uniform', 
                    Tspan=Tspan, 
                    components=gwb_components,
                    orf=orf, 
                    name = 'gw_bhb')

        else:
            s += common_red_noise_block(
                    psd='powerlaw', 
                    prior='log-uniform', 
                    Tspan=Tspan, 
                    components=gwb_components,
                    gamma_val=gamma_bhb, 
                    orf=orf, 
                    name = 'gw_bhb',
                    logmin=A_bhb_logmin, 
                    logmax=A_bhb_logmax)


    # add DM variations 
    dm_var = [hasattr(psr, 'dmx') for psr in psrs] # check if dmx parameters are present in pulsars objects

    if all(dm_var):
        pass
    elif not any(dm_var):
        s += dm_noise_block(
            gp_kernel='diag', 
            psd='powerlaw',
            prior='log-uniform',
            components=30, 
            gamma_val=None)
    else:
        sys.exit('ERROR: The convention for DM variation is not consistent across the PTA data.')


    # add new-physics signal
    if model:
        if hasattr(model, "signal"):
            signal = model.signal(**model.parameters)
            np_signal = deterministic_signals.Deterministic(signal, name=model.name)

            s += np_signal
        
        elif hasattr(model, "spectrum"):
            cpl_np = model.spectrum(**model.parameters)

            if corr:
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
        if 'NANOGrav' in p.flags['pta']:
            s2 = s + white_noise_block(
                vary=white_vary, inc_ecorr=True, tnequad=tnequad, select='backend')
            if '1713' in p.name and not any(dm_var):
                s3 = s2 + chrom.dm_exponential_dip(
                    tmin=54500, tmax=55000, idx=2, sign=False, name='dmexp_1')
                if p.toas.max() / const.day > 57850:
                    s3 += chrom.dm_exponential_dip(
                        tmin=57300, tmax=57850, idx=2, sign=False, name='dmexp_2')
                models.append(s3(p))
            else:
                models.append(s2(p))
        else:
            s4 = s + white_noise_block(
                vary=white_vary, inc_ecorr=False, tnequad=tnequad, select='backend')
            if '1713' in p.name and not any(dm_var):
                s5 = s4 + chrom.dm_exponential_dip(
                    tmin=54500, tmax=55000, idx=2, sign=False, name='dmexp_1')
                if p.toas.max() / const.day > 57850:
                    s5 += chrom.dm_exponential_dip(
                        tmin=57300, tmax=57850, idx=2, sign=False, name='dmexp_2')
                models.append(s5(p))
            else:
                models.append(s4(p))

    # set up PTA
    pta = signal_base.PTA(models)

    # set white noise parameters
    if noisedict is not None:
        pta.set_default_params(noisedict)

    return pta
    