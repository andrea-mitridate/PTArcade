import sys
from enterprise.signals import utils
from enterprise.signals import signal_base
from enterprise.signals import white_signals
from enterprise.signals import gp_signals
import enterprise.signals.parameter as parameter
from enterprise.signals import selections
from enterprise.signals import deterministic_signals

import numpy as np

def builder(model, psrs, noise_params, N_f_red, N_f_gwb, base_mod=True):

    tmin = [p.toas.min() for p in psrs]
    tmax = [p.toas.max() for p in psrs]
    Tspan = np.max(tmax) - np.min(tmin)

    # timing model
    tm = gp_signals.TimingModel(use_svd=True)

    s = tm

    # disperion measure 
    dm = [hasattr(psr, 'dmx') for psr in psrs]
    if any(dm):
        pass
    elif not any(dm):
        dm_log10_A = parameter.Uniform(-20, -11)
        dm_gamma = parameter.Uniform(0, 7)
        dm_pl = utils.powerlaw(log10_A=dm_log10_A, gamma=dm_gamma)
        dm_basis = utils.createfourierdesignmatrix_dm(nmodes=30, Tspan=Tspan)

        dmgp = gp_signals.BasisGP(dm_pl, dm_basis, name='dm_gp')

        s += dmgp
    else:
        sys.exit('The convention for the DM variations is not consistent across the PTA data.')


    # white noise parameters
    t2 = False
    tm = False

    for x in noise_params:
        if 't2equad' in x:
            t2 = True
        elif 'equad' in x:
            tm = True

    if t2 and tm:
        sys.exit('The convention for equad is not consistent across the PTA data.')

    if noise_params:
        # define selection by observing backend
        bkend = selections.Selection(selections.by_backend)
        bkend_ng = selections.Selection(selections.nanograv_backends)

        efac = parameter.Constant() 
        equad = parameter.Constant() 
        ecorr = parameter.Constant()

        ec = white_signals.EcorrKernelNoise(log10_ecorr=ecorr, selection=bkend_ng)

        if t2:
            ms = white_signals.MeasurementNoise(efac=efac, log10_t2equad=equad, selection=bkend)
            s += ms
        elif tm:
            ef = white_signals.MeasurementNoise(efac=efac, selection=bkend)
            eq = white_signals.TNEquadNoise(log10_tnequad=equad, selection=bkend)
            s += ef + eq

    else:
        efac = parameter.Uniform(0.01, 10.0)
        equad = parameter.Uniform(-8.5, -5.0)

        ms = white_signals.MeasurementNoise(efac=efac, log10_t2equad=equad)

        s += ms


    # red noise parameters
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

    s_ng = s + ec

    # intialize PTA
    mods = []
    for psr in psrs:    
        if 'NANOGrav' in psr.flags['pta']:
            mods.append(s_ng(psr))
        else:
            mods.append(s(psr)) 
    
    pta = signal_base.PTA(mods)

    if noise_params:
        pta.set_default_params(noise_params)

    return(pta)



    
