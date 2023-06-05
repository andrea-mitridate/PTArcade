# Overview

## Auxiliary functions/constants 

The `models_utils` module containts auxiliary functions and constatns that can help in the definition of the new-physics signal. Here is a brief summary of the content of this module.

### Physical constants

... under construction ...

### Auxiliary functions (under construction)

- `g_rho` is a function takes as argument float `T` representing the Universe temperature in GeV and return the number of relativistic degrees of freedom.

- `g_s` is a function takes as argument float `T` representing the Universe temperature in GeV and return the number of entropic relativistic degrees of freedom.

- `Gamma` implementation of a gamma distribution 

- `omega2cross` is a function that takes as input the spectrum of a GWB expressed $h^2\Omega(f,\ldots)$ and return a cross-power spectral density funtion $S(f,\ldots)$

- `spec_importer` takes the path to some tabulated expression for the gravitational wave spectrum and returns an interpolated function for it. The data need to be tab separated, and each column in the file needs to start with the name of the variable containted in that column. The column containing the frequency needs to be called `f` and the one containing the value of the GWB spectrum needs to be called `spectrum`.
