# Inputs

When [running PTArcade][run] the user can provide two input files:

- [Model file][model] · :octicons-alert-24: __Required__ – This file, passed via the `-m` input flag, containts the definition of the new-physics signal. In case of stochastic signals, this boils down to defining the GWB energy density per logarithmic frequency interval, $d\rho_{\textrm{GW}}/d\ln f$, as a fraction of the closure density:

    $$
    h^2\Omega_{\textrm{GW}}(f;\,\vec{\theta}) \equiv \frac{h^2}{\rho_c}\frac{d\rho_{\textrm{GW}}(f;\,\vec{\theta})}{d\ln f}\,.
    $$

    where $\vec{\theta}$ is the set of new physics parameters describing the signals. In case of deterministic signals, the user should define the timeseris of induced timing delays $s(t;\,\vec{\theta})$ in units of seconds.


- [Configuration file][config] · :octicons-plus-16: __Optional__ – In addition to the model file, the user can also pass a configuration file via the input flag `-c`. The configuration file is a simple Python file that allows the user to adjust several parameters of the run.

In the following we will discuss the details of both these inputs files. 

  [run]: ../getting_started/run.md
  [model]: model.md
  [config]: config.md
