# Search for Non Black Hole Binaries signals in NANOGrav data 

This code provides a simple interface to the ENTERPRISE analysis suite and allows for easy implementations of new-physics searches in NANOGrav data. 

The user can specify a new physics signal (either deterministic or stochastic), and the code will output Monte Carlo chains that can be used to reconstruct the model's parameter posterior distributions. 

# Installation 

1) Install `(mini)conda`, an environment management system for python, from [here](https://conda.io/projects/conda/en/latest/user-guide/install/index.html])
2)  

    > conda env create -f environment.yml
    > conda activate non-bhb-search

3) Unzip `pta_data.zip` folder located in './inputs/pta_data.zip`

# Input 

All the input files required for the code to run are stored in `/inputs`. There are two types of input files required by the code: PTA data (such as timing measurements, pulsar noise parameters, etc...) provided by one of the IPTA collaborations, and info files specified by the user (for example to specify the new-physics signal, numerical parameter of the Monte Carlo, etc...).
## Pulsar Data

The essential pulsar data needed by the code to run are the `.tim` and `.par` files which need to be stored in the `inputs/pta_data` directory. 

Additional pulsar data can be used by the code (as previously, if used they need to be stored in `inputs/pta_data`). Specifically:

- White noise parameters
- Prior distributions 

## User-specified inputs

These input files allow the user to specify the new-physics signal, and the `enterprise` parameters to use in the analaysis. These files are just python files, so any modifications to these can call other python modules/libraries, or create arrays of values algorithmically.


### Model info: `inputs/models/(model_info_file_name).py`
Specifies the new-physics signal, and it needs to be stored in `/inputs/models`. For deterministic signals, they must include a `signal(toas, ...)` function, which specifies the signal shape as a function of the time of arrivals, `toas`, and any other model parameter. For stochastic signals, the model file must include a `spectrum(f,...)` function that specifies the cross-power spectral density of the gravitational background. 

In addition to this, for both deterministic and stochastic signals, the model file needs to include the prior of the model parameters. 

In the model file it is also possible to specify:

- The name of the model through the variable `name`
- Whether or not the SMBHB signal should be added to the new physics signal by setting the variable `smbhb` to `True` or `False`
- Whether or not spatial correlation should be used in the analysis by setting the variable `corr` to `True` or `False`
- Groups of parameters that should be sampled together by adding their name to the list `group`


### Numerics/enterprise info: `inputs/numerics/(numerics_info_file_name).py`

The numerics info allow to specify enterprise parameters for the run:

- `pta_data` string that specifies the set of PTA data to use in the analysis. The user can choose among NG15 (NANOGrav 15yr data), NG12 (NANOGrav 12.5yr data), IPTA2 (IPTA DR2 data,). PTA data which are not present in the above list can also be used by passing as `pta_data` a dictionary containing the following informations
    -  `pta_data['psrs_data']`: string with the name of the pickle file containing the pulsar objects, or name of the folder containing the `.tim` and `.par` files
    - `pta_data['noise_data']`: string with the name of the folder or `.json` file containing the white noise parameters (optional, if the user does not want to use pre-derived white noise parameter this parameter can be set to `None` (object not string). In this case the code will assume flat priors for the white noise parameters and sample them in the mcmc run.)
    - `pta_data['emp_dist']`: string with the name of the empirical distribution file. If the user do not want to use empirical distributions, this value can be set to `None` (object not string).
    All the files specified in the key values of this dictionary need to be stored in `inputs/pta_data`.
- `mod_sel` boolean variable that specifies whether or not the model should be compared with the SMBHB signal
- `out_dir` name of the directory where to save the output files
- `N_samples` number of Monte Carlo trials 
- `red_components` number of frequency components for the pulsar intrinsic red noise 
- `gwb_components` number of frequency components for all the stochastic common processes 
- `A_bhb_logmin` lower limit for the prior of the bhb signal amplitude. If set to None -18 is used
- `A_bhb_logmin` upper limit for the prior of the bhb signal amplitude. If set to None -14 is used
- `gamma_bhb` spectral index for the bhb singal. If set to None it's varied between [0, 7].

# How to use

    > python sampler.py -m (model info file).py -n (numeri info file).py -c (chain number)


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
