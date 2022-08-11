# Search for Non Black Hole Binaries signals in NANOGrav data 

This code provides a simple interface to the ENTERPRISE analysis suite and allows for easy implementations of new-physics searches in NANOGrav data. 

The user can specify a new physics signal (either deterministic or stochastic), and the code will output Monte Carlo chains that can be used to reconstruct the model's parameter posterior distributions. 

# Installation 

1) Install `(mini)conda`, an environment management system for python, from [here](https://conda.io/projects/conda/en/latest/user-guide/install/index.html])
2)  

    > conda env create -f environment.yml
    > conda activate <env>

# Input 

All the input files required for the code to run are stored in `/inputs`. There are two types of input files required by the code: PTA data (such as timing measurements, pulsar noise parameters, etc...) provided by one of the IPTA collaborations, and info files specified by the user (for example to specify the new-physics signal, numerical parameter of the Monte Carlo, etc...).
## Pulsar Data

The essential pulsar data needed by the code to run are the `.tim` and `.par` files which need to be stored in the `inputs/pta/pta_data` directory. 

Additional pulsar data can be used by the code (as previously, if used they need to be stored in `inputs/pta/pta_data`). Specifically:

- White noise parameters
- Prior distributions 

## User-specified inputs

These input files allow the user to specify the new-physics signal, the PTA data to use in the analysis, and the numerical parameters of the run. These files are just python files, so any modifications to these can call other python modules/libraries, or create arrays of values algorithmically.


### Model info: `inputs/models/(model_info_file_name).py`
Specifies the new-physics signal, and it needs to be stored in `/inputs/models`. For deterministic signals, they must include a `signal(toas, ...)` function, which specifies the signal shape as a function of the time of arrivals, `toas`, and any other model parameter. For stochastic signals, the model file must include a `spectrum(f,...)` function that specifies the cross-power spectral density of the gravitational background. 

In addition to this, for both deterministic and stochastic signals, the model file needs to include the prior of the model parameters. 

In the model file it is also possible to specify:

- The name of the model through the variable `name`
- Whether or not the model should be compared with the SMBHB signal, by setting the variable `mod_sel` to `True` or `False`
- Whether or not the SMBHB signal should be added to the new physics signal by setting the variable `smbhb` to `True` or `False`
- Whether or not spatial correlation should be used in the analysis by setting the variable `corr` to `True` or `False`
- Groups of parameters that should be sampled together by adding their name to the list `group`


### Numerics info: `inputs/numerics/(numerics_info_file_name).py`

The numerics info file specifies the parameters of the Monte Carlo run:

- `out_dir` name of the directory where to save the output files
- `N_samples` number of Monte Carlo trials 
- `N_f_red` number of frequency components for the pulsar intrinsic red noise 
- `N_f_gwb` number of frequency components for all the stochastic common processes 


### PTA info: `inputs/pta/(pta_info_file_name).py`

Specifies which pta data in the `inputs/pta/pta_data` folder need to be used for the analysis. 

- `psr_data` name of the pickle file containing the pulsar objects, or name of the folder containing the `.tim` and `.par` files
- `noise_data` name of the folder or `.json` file containing the white noise parameters (optional, if the user does not want to use pre-derived white noise parameter this parameter can be set to an empty string. In this case the code will assume flat priors for the white noise parameters and sample them in the mcmc run.)
- `emp_dist` name of the empirical distribution file (required if the pulsar objects are specified with `.tim` and `.par` files)

All the files specified in the PTA info file need to be stored in `inputs/pta/pta_data`.

# How to use

    > python sampler.py -m (model info file).py -n (numeri info file).py -p (PTA info file).py - c (chain number)


# Output

The output is stored in the `numeric_info.out_dir/model_info.name/chain#`, and it consists of the usual ENTERPRISE output. 


# Auxiliary functions/constants 

The `models_utils` module containts auxiliary functions and constatns that can help in the definition of the new-physics signal. Here is a brief summary of the content of this module.

## Physical constants

... under construction ...

## Auxiliary functions (under construction)

- `g_rho` is a function takes as argument float `T` representing the Universe temperature in GeV and return the number of relativistic degrees of freedom.

- `g_s` is a function takes as argument float `T` representing the Universe temperature in GeV and return the number of entropic relativistic degrees of freedom.

- `Gamma` implementation of a gamma distribution 

- `omega2cross` is a function that takes as input the spectrum of a GWB expressed $h^2\Omega(f,\ldots)$ and return a cross-power spectral density funtion $S(f,\ldots)$

- `spec_importer` takes the path to some tabulated expression for the gravitational wave spectrum and returns an interpolated function for it. The data need to be tab separated, and each column in the file needs to start with the name of the variable containted in that column. The column containing the frequency needs to be called `f` and the one containing the value of the GWB spectrum needs to be called `spectrum`.
