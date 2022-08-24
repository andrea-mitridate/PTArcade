import sys
from enterprise.signals import utils
from enterprise.signals import signal_base
from enterprise.signals import white_signals
from enterprise.signals import gp_signals
import enterprise.signals.parameter as parameter
from enterprise.signals import selections
from enterprise.signals import deterministic_signals
from enterprise_extensions import model_utils
from enterprise import constants as const
from enterprise_extensions import chromatic as chrom

from enterprise_extensions.blocks import (chromatic_noise_block,
                                          common_red_noise_block,
                                          dm_noise_block, red_noise_block,
                                          white_noise_block)

import numpy as np

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
    :gamma_bhb: fixed common bhb process spectral index value. If set to
        None we vary the spectral index over the range [0, 7].
        [default = None]
    :param A_bhb_logmin: specifies lower prior on the log amplitude of the bhb
        common process. If set to None, -18 is used.
        [default = None] 
    :param A_bhb_logmax: specifies upper prior on the log amplitude of the bhb
        common process. If set to None, -14 is used if gamma_bhb = 13/3, -11 is
        used otherwise.
        [default = None]
    :red_components: number of frequency components for the intrinsic
        red noise. 
        [default = 30]
    :gwb_components: number of frequency components for the common processes.
        [default = 14]
    """

    # timing model
    tm = gp_signals.TimingModel(use_svd=True)
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
        
        s += common_red_noise_block(
            psd='powerlaw', 
            prior='lof-uniform', 
            Tspan=Tspan, 
            components=gwb_components,
            gamma_val=gamma_bhb, 
            orf=orf, 
            name = 'gw_bhb',
            logmin=A_bhb_logmin, 
            logmax=A_bhb_logmax)


    # add DM variations 
    dm_var = [hasattr(psr, 'dmx') for psr in psrs]

    if any(dm_var):
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
                    name=model.name, 
                    Tspan=Tspan)
            else:
                np_gwb = gp_signals.FourierBasisGP(
                    spectrum=cpl_np, 
                    components=gwb_components,
                    Tspan=Tspan, 
                    name=model.name)

            s += np_gwb


    # add white-noise, and act on psr objects 
    models = []
 
    if noisedict:
        white_vary = False
        tnequad = tnequad_conv(noisedict)
    else:
        white_vary = True    

    for p in psrs:
        if 'NANOGrav' in p.flags['pta']:
            s2 = s + white_noise_block(
                vary=white_vary, inc_ecorr=True, tnequad=tnequad, select='backend')
            if '1713' in p.name and dm_var:
                tmin = p.toas.min() / const.day
                tmax = p.toas.max() / const.day
                s3 = s2 + chrom.dm_exponential_dip(
                    tmin=tmin, tmax=tmax, idx=2, sign=False, name='dmexp')
                models.append(s3(p))
            else:
                models.append(s2(p))
        else:
            s4 = s + white_noise_block(
                vary=white_vary, inc_ecorr=False, tnequad=tnequad, select='backend')
            if '1713' in p.name and dm_var:
                tmin = p.toas.min() / const.day
                tmax = p.toas.max() / const.day
                s5 = s4 + chrom.dm_exponential_dip(
                    tmin=tmin, tmax=tmax, idx=2, sign=False, name='dmexp')
                models.append(s5(p))
            else:
                models.append(s4(p))


    # set up PTA
    pta = signal_base.PTA(models)


    # set white noise parameters
    if noisedict:
        pta.set_default_params(noisedict)

    return pta