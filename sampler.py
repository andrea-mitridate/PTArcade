import os
import sys
import optparse
import numpy as np
from enterprise_extensions import hypermodel

import src.pta_importer as pta_importer
import src.signal_builder as signal_builder
import src.input_handler as input_handler


print('\n--- Starting to run ---\n\n')

###############################################################################
# load inputs
###############################################################################

# parse command line inputs
input_options, cmd_input_okay = input_handler.get_cmdline_arguments()

if not cmd_input_okay:

    print('ERROR:')
    print("\t- Models info file\n"+
    "\t- Numerics info file\n"+
    "must be present. These are added with the -m, -n input flags. Add -h (--help) flags for more help.")

    sys.exit()

inputs = input_handler.load_inputs(input_options)

###############################################################################
# load pulsars and noise parameters
###############################################################################
print('--- Loading Pulsars and noise data ... ---\n')


# import pta data
psrs, noise_params, emp_dist = pta_importer.pta_data_importer(inputs['numerics'].pta_data)

print('--- Done loading Pulsars and noise data. ---\n\n')

###############################################################################
# define models and initialize PTA
###############################################################################
print('--- Initializing PTA ... ---\n')

pta = {}

pta[0] = signal_builder.builder(
    psrs=psrs, 
    model=inputs['model'], 
    noisedict=noise_params,
    gamma_bhb=inputs['numerics'].gamma_bhb,
    A_bhb_logmin=inputs['numerics'].A_bhb_logmin,
    A_bhb_logmax=inputs['numerics'].A_bhb_logmax,
    corr=inputs['numerics'].corr,
    red_components=inputs["numerics"].red_components, 
    gwb_components=inputs["numerics"].gwb_components)

if inputs["numerics"].mod_sel:
    pta[1] = pta[0]

    pta[0] = signal_builder.builder(
        psrs=psrs, 
        model=None, 
        noisedict=noise_params,
        gamma_bhb=inputs['numerics'].gamma_bhb,
        A_bhb_logmin=inputs['numerics'].A_bhb_logmin,
        A_bhb_logmax=inputs['numerics'].A_bhb_logmax,
        corr=inputs['numerics'].corr,
        red_components=inputs["numerics"].red_components, 
        gwb_components=inputs["numerics"].gwb_components)


print('--- Done initializing PTA. ---\n\n')

###############################################################################
# define sampler and sample
###############################################################################

out_dir = os.path.join(
    inputs["numerics"].out_dir, inputs["model"].name, f'chain_{input_options["c"]}')

super_model = hypermodel.HyperModel(pta)
sampler = super_model.setup_sampler(
    resume=False,
    outdir=out_dir,
    sample_nmodel=inputs["numerics"].mod_sel,
    empirical_distr=emp_dist)

x0 = super_model.initial_sample()

if inputs["model"].group:
    pars = np.loadtxt(out_dir + '/pars.txt', dtype=np.unicode_)

    idx_params = [list(pars).index(pp) for pp in pars if pp in inputs["model"].group]
    [sampler.groups.append(idx_params) for _ in range(5)]
    groups = sampler.groups

    sampler = super_model.setup_sampler(
        resume=False,
        outdir=out_dir,
        sample_nmodel=inputs["numerics"].mod_sel, 
        groups=groups, 
        empirical_distr=emp_dist)

print('--- Starting to sample... ---\n')

sampler.sample(
    x0, 
    inputs["numerics"].N_samples,
    SCAMweight=inputs['numerics'].scam_weight,
    AMweight=inputs['numerics'].am_weight,
    DEweight=inputs['numerics'].de_weight)

print('--- Done sampling. ---\n\n')
