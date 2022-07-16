import os
import sys
import optparse
import numpy as np
from enterprise_extensions import hypermodel

import src.pta_importer as pta_importer
import src.signals as signals
import src.input_hander as input_handler


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
    "\t- PTA input file\n\n"+
    "must be present. These are added with the -m, -n, -c input flags. Add -h (--help) flags for more help.")

    sys.exit()

inputs = input_handler.load_inputs(input_options)

###############################################################################
# load pulsars and noise parameters
###############################################################################
print('--- Loading Pulsars and noise data ... ---\n')
        
# import pulsars in ENTERPRISE
psrs = pta_importer.get_pulsars(inputs["pta_params"].psr_data, ephemeris=inputs["pta_params"].ephem)

# import noise parameters
if inputs["pta_params"].noise_data:
    noise_params = pta_importer.get_wn(inputs["pta_params"].noise_data)
else:
    noise_params = False

print('--- Done loading Pulsars and noise data. ---\n\n')

###############################################################################
# define models and initialize PTA
###############################################################################
print('--- Initializing PTA ... ---\n')

pta = {}

if inputs["model"].mod_sel:
    pta[0] = signals.builder(model=inputs["model"], psrs=psrs, noise_params=noise_params,
                N_f_red=inputs["numerics"].N_f_red, N_f_gwb=inputs["numerics"].N_f_gwb)

    pta[1] = signals.builder(model=inputs["model"], psrs=psrs, noise_params=noise_params,
            N_f_red=inputs["numerics"].N_f_red, N_f_gwb=inputs["numerics"].N_f_gwb, base_mod=False)

else:
    pta[0] = signals.builder(model=inputs["model"], psrs=psrs, noise_params=noise_params,
            N_f_red=inputs["numerics"].N_f_red, N_f_gwb=inputs["numerics"].N_f_gwb, base_mod=False)

print('--- Done initializing PTA. ---\n\n')

###############################################################################
# define sampler and sample
###############################################################################

out_dir = f'{inputs["numerics"].out_dir}/{inputs["model"].name}/chain_' + input_options['c']

super_model = hypermodel.HyperModel(pta)

sampler = super_model.setup_sampler(resume=False, outdir=out_dir, sample_nmodel=inputs["model"].mod_sel,
    empirical_distr=pta_importer.pta_dat_dir + inputs["pta_params"].emp_dist)

x0 = super_model.initial_sample()

if inputs["model"].group:
    # load the list of parameters name
    pars = np.loadtxt(out_dir + '/pars.txt', dtype=np.unicode_)

    idx_params = [list(pars).index(pp) for pp in pars if pp in inputs["model"].group]
    [sampler.groups.append(idx_params) for _ in range(5)] # chosen 5 here so that this group is visited alot
    groups = sampler.groups

    sampler = super_model.setup_sampler(resume=False, outdir=out_dir, sample_nmodel=inputs["model"].mod_sel, 
    groups=groups, empirical_distr=pta_importer.pta_dat_dir + inputs["pta_params"].emp_dist)

print('--- Starting to sample... ---\n')

sampler.sample(x0, inputs["numerics"].N_samples, SCAMweight=30, AMweight=15, DEweight=50)

print('--- Done sampling. ---\n\n')
