# Getting started  
PTArcade is a wrapper of [ENTERPRISE] that streamlines the implementation of
Bayesian inference analyes for new physics models using PTA data.

Confused already? Let's try to be more concrete. Say you have a new physics model 
that produces a gravitational background with a relic abundance given by
$\Omega_{\textrm{GW}}(f;\,\vec{\theta})$, where $\vec{\theta}$ is a set of parameters
describing the signal. You might now be interested in konwing if there are regions in 
the $\vec{\theta}$ parameter space that could reproduced the signal observed in PTA
experiments[^1], or that are excluded by PTA data. PTArcade will allow you to answer these
question in under 10 minutes (plus computation time) by using real PTA data, and the same
statistical tools used by PTAs collaborations.

[^1]: See [here][NG15] for more details about the PTA signal.

The only thing that is asked to the user is to define the GWB produced by their model, 
and the prior distribution of the model parameters. This is done via what we call a
[_model file_][model]. Let's consider the specific example where the

$$
h^2\Omega_{\textrm{GW}}(f) = \frac{A_*}{f/f_* + f_*/f}
$$

in this case the model file will look something like this

``` py 
from ptarcade import parameter

parameters = {
            'log_A_star' : parameter.Uniform(-14, -6), # (1)!
            'log_f_star' : parameter.Uniform(-10, -6) 
            } 

def S(x):
    return 1 / (1/x + x)

def spectrum(f, log_A_star, log_f_star): # (2)!
    A_star = 10**log_A_star
    f_star = 10**log_f_star
    
    return A_star * S(f/f_star) # (3)!
```

1. The parameters priors are defined via a dictionary named `parameters`.
The keys of this dictionary will be the name of the model parameters, and
the values are parameters objects specifying the parameters prior distribution.

2. The GWB spectrum is defined via the `spectrum` function. The first argument
of this function has to be called `f` and is supposed to be an array of frequencies
(in unit of Hz) at which the spectrum will be evaluated. The remaining parameter
should be named as the keys of the `parameters` dictionary. 

3. For any given set of new-physics parameters (in this example `log_A_star` and 
`log_f_star`, the `spectrum` function should return an array which contains the value of
 $h^2\Omega_{\textrm{GW}}$ evaluated at those parameter values and at all the 
 frequencies containes in the `f` list.

Once you have defined the model file, you can feed it to PTArcade by running in a 
terminal 

```shell
ptarcade -m ./model_file.py # (1)!
```

1. The argument passed to the `-m` input flag should be the path to the model file. 
Here we are assuming that the model file is named `model_file.py` and is located in
the folder from which we are running PTArcade. 

This will implement and run the analysis whose results are Monte Carlo chains that
can be used to derive posterior distributions for the parameters of our model. For
the example taht we are considering here this is the result:

!!! info "Deterministic signals"
    PTArcade can also be used with deterministic signals. In this case, the user will
    have to specify the signal timeseries instead of the signal power-spectrum in the
    model file. See [here][model] for more details. 

After this high level summary of what PTArcade can do, in the next sections we will
discuss:

- how to install PTArcade ([locally][local_install], or [on a cluster][hpc_install])
- [how to define a model file][model]
- how to change the parameters of PTArcade runs (PTA dataset, number of MC points, ...)
using a [configuration file][config]
- the structure of the [PTArcade outptut][output]
- PTArcade utilities that can [help in constructing model files][model_utils] or
 [analyzing][chain_utils] and [plotting][plot_utils] the MC chains.

!!! note "How to cite PTArcade"

    If you use PTArcade in your work, please cite

    ```
    @article{andrea mitridate_2023,
        title={PTArcade},
        DOI={10.5281/zenodo.7876430},
        publisher={Zenodo},
        author={Andrea Mitridate}, 
        year={2023}, 
        month={Apr},
        copyright = {Open Access}}

    @article{mitridate...}    
    ```

  [ENTERPRISE]: https://github.com/nanograv/enterprise
  [local_install]: local_install.md
  [hpc_install]: hpc_install.md
  [model]: ../inputs/model.md
  [config]: ../inputs/config.md
  [output]: ../outputs.md
  [model_utils]: ../utils/model_utils.md
  [chain_utils]: ../utils/chain_utils.md
  [plot_utils]: ../utils/plot_utils.md
  [NG15]: https://aas.com