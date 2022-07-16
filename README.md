# Search for Non Black Holes Binary signal in NANOGrav data 

The code provides a simple interface to the ENTERPRISE analysis suite, and allows for an easy implementation of new-physics searches in NANOGrav data. 

The user can specify a new physics signal (either deterministic or stochastic) and the code will outpout Monte Carlo chains that can be used to reconstruct the model's parameters posterior distributions. 

# Installation 

1) Install `(mini)conda`, an environment management system for python, from [here](https://conda.io/projects/conda/en/latest/user-guide/install/index.html])
2)  

    > conda create --name <env> --file requirements.txt
    > conda activate <env>

# Input 

All the inputs files required for the code to run are stored in `/inputs`. There are two types of input files required by the code: pta data (such as timing measurments, pulsar noise parameters, ecc...) provided by one of the IPTA collaborations, and info files specified by the user (for example to specify the new-physics signal, numerical paramter of the Monte Carlo, ecc...). The input files are just python files, so any modifications to these can call other python modules/libraries, or create arrays of values algorithmically.

## Pulsar Data

The essential pulsar data needed by the code to run are the `.tim` and `.par` files which need to be stored in the `inputs/pta/pta_data` directory. 

Additional pulsar data can be used by the code (as previously, if used they need to be stored in `inputs/pta/pta_data`). Specifically:

- White noise parameters
- Prior distributions 

## User specified inputs

This input files allow the user to specify the new-physics signal, the pta data to use in the analysis, and the numerical parameters of the run. 

### Model info: `inputs/models/(model_info_file_name)`
Specifies the new-physics signal, and they need to be stored in `/inputs/models`. For deterministic signal they must include a `signal(toas, ...)` function, which specify the signal shape as a function of the time of arriavals, `toas`, and any other model parameter. For stochastic signas, the model file must include a `spectrum(f,...)` function that specify the cross-power spectral density of the gravitational background. 

In addition to this, for both deterministic and stochastic signals, the model file needs to include the prior of the model parameters. 

In the model file it is also possible to specify:

- The name of the model through the variable `name`
- Whether or not the model should be compared with the SMBHB signal, by setting the variable `mod_sel` to `True` or `False`
- Whether of not the SMBHB signal should be added to the new physics signal by setting the variable `smbhb` to `True` or `False`
- Whether or not spatial correlation should be used in the analysis by setting the variable `corr` to `True` or `False`
- Groups of parameters that should be sampled together by adding their name to the list `group`


### Numerics info: `inputs/numerics/(numerics_info_file_name)`

The numerics file specify the parameters of the Monte Carlo run:

- `out_dir` name of the directory where to save the ouptut files
- `N_samples` number of Monte Carlo trials 
- `N_f_red` number of frequency components for the pulsar intrinsic red noise 
- `N_f_gwb` number of frequency components for all the stochastic common processes 


### PTA info: `inputs/pta/(pta_info_file_name)`

Spefies which pta data in the `inputs/pta/pta_data` folder to use for the analysis. 

- `psr_data` name of the folder or pickle file containing the `.tim` and `.par` files
- `noise_data` name of the folder of `.json` file containing the white noise parameters (optional)
- `emp_dist` name of the empirical distribution file (optional)

All the files specified in the PTA info file needs to be stored in `inputs/pta/pta_data`.

# How to use

> python sampler.py -m (model info file) -n (numeri info file) -p (PTA info file) - c (chain number)


# Output

The output is store in the `numeric_info.out_dir/model_info.name/chain#`, and it consists of the ususal ENTERPRISE output. 