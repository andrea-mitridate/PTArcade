from enterprise.signals import utils
from enterprise.signals import signal_base
from enterprise.signals import white_signals
from enterprise.signals import gp_signals
import enterprise.signals.parameter as parameter
from enterprise.signals import selections
from enterprise.signals import deterministic_signals

import numpy as np

def builder(model, psrs, noise_params, N_f_red, N_f_gwb, base_mod=True):

    # timing model
    tm = gp_signals.TimingModel(use_svd=True)

    s = tm

    # white noise parameters
    if noise_params:
        # define selection by observing backend
        selection = selections.Selection(selections.by_backend)

        efac = parameter.Constant() 
        equad = parameter.Constant() 
        ecorr = parameter.Constant()

        ms = white_signals.MeasurementNoise(efac=efac, log10_t2equad=equad, selection=selection)
        ec = white_signals.EcorrKernelNoise(log10_ecorr=ecorr, selection=selection)

        s += ms + ec

    else:
        efac = parameter.Uniform(0.01, 10.0)
        equad = parameter.Uniform(-8.5, -5.0)

        ms = white_signals.MeasurementNoise(efac=efac, log10_t2equad=equad)

        s += ms


    # red noise parameters
    tmin = [p.toas.min() for p in psrs]
    tmax = [p.toas.max() for p in psrs]
    Tspan = np.max(tmax) - np.min(tmin)

    log10_A_red = parameter.Uniform(-20, -11)
    gamma_red = parameter.Uniform(0, 7)

    pl = utils.powerlaw(log10_A=log10_A_red, gamma=gamma_red)
    rn = gp_signals.FourierBasisGP(spectrum=pl, components=N_f_red, Tspan=Tspan)

    s += rn


    # GWB from SMBHB
    if model.smbhb:
        # find the maximum time span to set GW frequency sampling
        tmin = [p.toas.min() for p in psrs]
        tmax = [p.toas.max() for p in psrs]
        Tspan = np.max(tmax) - np.min(tmin)

        log10_A_smbhb = parameter.Uniform(-18,-14)('log10_A_gw')
        gamma_smbhb = parameter.Constant(4.33)('gamma_gw')

        cpl = utils.powerlaw(log10_A=log10_A_smbhb, gamma=gamma_smbhb)

        if model.corr:
            orf = utils.hd_orf()
            smbhb = gp_signals.FourierBasisCommonGP(spectrum=cpl, orf=orf, components=N_f_gwb, name='smbhb', Tspan=Tspan)
        else:
            smbhb = gp_signals.FourierBasisGP(spectrum=cpl, components=N_f_gwb, Tspan=Tspan, name='smbhb')

        s += smbhb


    # new physics signal
    if hasattr(model, "signal") and not base_mod:
        signal = model.signal(**model.parameters)
        np_signal = deterministic_signals.Deterministic(signal, name=model.name)

        s += np_signal
    
    elif hasattr(model, "spectrum") and not base_mod:
        cpl_np = model.spectrum(**model.parameters)

        if model.corr:
            orf = utils.hd_orf()
            np_gwb = gp_signals.FourierBasisCommonGP(spectrum=cpl_np, orf=orf, components=N_f_gwb, name=model.name, Tspan=Tspan)
        else:
            np_gwb = gp_signals.FourierBasisGP(spectrum=cpl_np, components=N_f_gwb, Tspan=Tspan, name=model.name)

        s += np_gwb


    # intialize PTA
    mods = []
    for psr in psrs:    
        mods.append(s(psr)) 
    
    pta = signal_base.PTA(mods)

    if noise_params:
        pta.set_default_params(noise_params)

    return(pta)



    
