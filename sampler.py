import os
import sys

import warnings
from astropy.utils.exceptions import AstropyDeprecationWarning
warnings.filterwarnings('ignore', category=AstropyDeprecationWarning)

from enterprise_extensions import hypermodel

import src.pta_importer as pta_importer
import src.signal_builder as signal_builder
import src.input_handler as input_handler

import time
import platform

try:
    import cpuinfo
    def cpu_model():
        return cpuinfo.get_cpu_info()["brand_raw"]

except ModuleNotFoundError:
    def cpu_model():
        return "unknown CPU (for better info install py-cpuinfo)"

print('\n--- Starting to run ---\n')
print("\tNode", platform.node(), cpu_model(),"\n");

start_cpu = time.process_time()
start_real = time.perf_counter()

###############################################################################
# load inputs
###############################################################################

# parse command line inputs
input_options, cmd_input_okay = input_handler.get_cmdline_arguments()

if not cmd_input_okay:

    print('ERROR:')
    print("\t- Model and configuration files must be present\n"+
    "\t- These are added with the -m, -c input flags. Add -h (--help) flags for more help.")

    sys.exit()

inputs = input_handler.load_inputs(input_options)

input_handler.check_config(inputs['config'])

###############################################################################
# load pulsars and noise parameters
###############################################################################
print('--- Loading Pulsars and noise data ... ---\n')

# import pta data
psrs, noise_params, emp_dist = pta_importer.pta_data_importer(inputs['config'].pta_data)

print(f"\t loaded {len(psrs)} pulsars\n")

input_handler.check_model(
    model=inputs['model'],
    psrs=psrs,
    red_components=inputs['config'].red_components,
    gwb_components=inputs['config'].gwb_components)

print('--- Done loading Pulsars and noise data. ---\n')

###############################################################################
# define models and initialize PTA
###############################################################################
print('--- Initializing PTA ... ---\n')

pta = {}

pta[0] = signal_builder.builder(
    psrs=psrs, 
    model=inputs['model'], 
    noisedict=noise_params,
    pta_dataset=inputs['config'].pta_data,
    bhb_th_prior=inputs['config'].bhb_th_prior,
    gamma_bhb=inputs['config'].gamma_bhb,
    A_bhb_logmin=inputs['config'].A_bhb_logmin,
    A_bhb_logmax=inputs['config'].A_bhb_logmax,
    corr=inputs['config'].corr,
    red_components=inputs["config"].red_components, 
    gwb_components=inputs["config"].gwb_components)

if inputs["config"].mod_sel:
    pta[1] = pta[0]

    pta[0] = signal_builder.builder(
        psrs=psrs, 
        model=None, 
        noisedict=noise_params,
        pta_dataset=inputs['config'].pta_data,
        bhb_th_prior=inputs['config'].bhb_th_prior,
        gamma_bhb=inputs['config'].gamma_bhb,
        A_bhb_logmin=inputs['config'].A_bhb_logmin,
        A_bhb_logmax=inputs['config'].A_bhb_logmax,
        corr=inputs['config'].corr,
        red_components=inputs["config"].red_components, 
        gwb_components=inputs["config"].gwb_components)


print('--- Done initializing PTA. ---\n\n')

###############################################################################
# define sampler and sample
###############################################################################

out_dir = os.path.join(
    inputs["config"].out_dir, inputs["model"].name, f'chain_{input_options["n"]}')

super_model = hypermodel.HyperModel(pta)

groups = signal_builder.unique_sampling_groups(super_model)

if inputs["model"].group:
    idx_params = [super_model.param_names.index(pp) for pp in inputs["model"].group]
    [groups.append(idx_params) for _ in range(5)]

# add nmodel index to group structure
groups.extend([[len(super_model.param_names)-1]])

sampler = super_model.setup_sampler(
    resume=inputs["config"].resume,
    outdir=out_dir,
    sample_nmodel=inputs["config"].mod_sel,
    groups=groups,
    empirical_distr=emp_dist)

x0 = super_model.initial_sample()

super_model.get_lnlikelihood(x0) # Cache now to make timing more accurate

print("Setup times (including first sample) {:.2f} seconds real, {:.2f} seconds CPU\n".format(
    time.perf_counter()-start_real, time.process_time()-start_cpu));
start_cpu = time.process_time()
start_real = time.perf_counter()

N_samples = inputs["config"].N_samples

print(f'--- Starting to sample {N_samples} samples... ---\n')

sampler.sample(
    x0, 
    N_samples,
    SCAMweight=inputs['config'].scam_weight,
    AMweight=inputs['config'].am_weight,
    DEweight=inputs['config'].de_weight)

print('--- Done sampling. ---')

real_time = time.perf_counter()-start_real
cpu_time = time.process_time()-start_cpu
print("Sampling times {:.2f} seconds real =  {:.4f} s/sample, {:.2f} seconds CPU =  {:.4f} s/sample\n".format(
    real_time, real_time/N_samples, cpu_time, cpu_time/N_samples))
