import os
import sys
import optparse
import numpy as np
from enterprise_extensions import hypermodel

import src.pta_importer as pta_importer
import src.signal_builder as signal_builder
import src.input_handler as input_handler

import time
import timeit
import platform


from pstats import SortKey
import cProfile
import pstats

try:
    import cpuinfo
    def cpu_model():
        return cpuinfo.get_cpu_info()["brand_raw"]

except ModuleNotFoundError:
    def cpu_model():
        return "unknown CPU (for better info install py-cpuinfo)"

print('\n--- Starting to run ---')
print("Node", platform.node(), cpu_model(),"\n");

start_cpu = time.process_time()
start_real = time.perf_counter()

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

print('--- Done loading Pulsars and noise data. ---')
print(len(psrs), "pulsars\n")

###############################################################################
# define models and initialize PTA
###############################################################################
print('--- Initializing PTA ... ---\n')

pta = {}

pta[0] = signal_builder.builder(
    psrs=psrs, 
    model=inputs['model'], 
    noisedict=noise_params,
    pta_dataset=inputs['numerics'].pta_data,
    bhb_th_prior=inputs['numerics'].bhb_th_prior,
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
        pta_dataset=inputs['numerics'].pta_data,
        bhb_th_prior=inputs['numerics'].bhb_th_prior,
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

groups = signal_builder.unique_sampling_groups(super_model)

if inputs["model"].group:
    idx_params = [super_model.param_names.index(pp) for pp in inputs["model"].group]
    [groups.append(idx_params) for _ in range(5)]

# add nmodel index to group structure
groups.extend([[len(super_model.param_names)-1]])

sampler = super_model.setup_sampler(
    resume=inputs["numerics"].resume,
    outdir=out_dir,
    sample_nmodel=inputs["numerics"].mod_sel,
    groups=groups,
    empirical_distr=emp_dist)

x0 = super_model.initial_sample()

super_model.get_lnlikelihood(x0) # Cache now to make timing more accurate

print("Setup times (including first sample) {:.2f} seconds real, {:.2f} seconds CPU".format(
    time.perf_counter()-start_real, time.process_time()-start_cpu));
start_cpu = time.process_time()
start_real = time.perf_counter()

print('--- Starting to sample... ---')

N_samples = inputs["numerics"].N_samples

print(N_samples, "samples\n")

sampler.sample(
    x0, 
    inputs["numerics"].N_samples,
    SCAMweight=inputs['numerics'].scam_weight,
    AMweight=inputs['numerics'].am_weight,
    DEweight=inputs['numerics'].de_weight)

print('--- Done sampling. ---\n\n')
