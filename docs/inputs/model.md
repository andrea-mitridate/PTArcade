The model file is a simple Python file that, at minimum, needs to contain the following informaiton:

* The [names and prior distributions][priors] of the new physics signal parameters.
* A parametrized form of the new physics signal, which can either be [stochastic][spectrum]
    (and parametrized via its power spectrum), or [deterministic][signal] (and parametrized
    as a timeseries).

In the following we will explain how these two quantities can be defined in the model file. 

  [priors]: #priors
  [spectrum]: #stochastic-signals
  [signal]: #deterministic-signals

## Priors
The priors for the signal parameters are defined via the `parameters` dictionary in the model file. The keys of this dictionary must be strings, which will be used as names of the model parameters. The values of this dictionary are `parameter` objects ...

??? example "Priors example"
    For example, the `parameters` dictionary of a model described by the parameters $a$ and $b$ which are common among all the pulsars will look like this for different choices of the priros:

    === "Uniform Priors"

        ``` py
        parameters = {'a' : parameter.Uniform(0,1), 'b' : parameter.Uniform(0,1)} # (1)!
        ```

        1.  In this case we have chosen uniform priors in the range [0,1] for both
            parameters.

    === "Normal Priors"

        ``` py
        parameters = {'a' : parameter.Normal(1,1), 'b' : parameter.Normal(1,1)} # (1)!
        ```
        
        1.  In this case we have chosen normal priors with unit mean and variance for
            both parameters.
    === "Exponential Priors"

        ``` py
        parameters = {'a' : parameter.Normal(1,1), 'b' : parameter.Normal(1,1)} # (1)!
        ```

        1.  In this case we have assumed 

## Stochastic signals
Stochastic signals are defined via the `spectrum` function in the model file. Its first parameter should be named `f` and it's supposed to be a [NumPy array][numpy] containing the frequencies (in unit of Hz) at which the spectrum is evaluated. The remaining parameters should match the ones defined in the `parameters` dictionary. The `spectrum` function should return a [NumPy array][numpy] with the same dimensions of `f` containing the value of $h^2\Omega_{\scriptscriptstyle\mathrm{GW}}$ at each of the frequencies contained in `f`.

??? example "Stochastic signal example"
    For example, the `spectrum` function for a model with 

    $$
    h^2\Omega_{\rm GW}(f) = 
    $$

    is given by

    ``` py
    def spectrum(f, a, b):

        return 
    ```
  
  [numpy]: https://numpy.org/doc/stable/reference/generated/numpy.array.html

## Deterministic signals
Deterministic signals are defined via the `signal` function in the model file. The first parameter of this function should be named `toas` and it's supposed to be a [NumPy array][numpy] containing the time of arrivals (TOAs) (in units of seconds) at which the deterministic signal will be evaluated. The remaining parameters should match the ones defined in the `parameters` dictionary. The `signal` function should return a [NumPy array][numpy] with the same dimensions of `toas` containing the value of the induced  shift for each TOA contained in `toas`.

??? example "Deterministic signal example"
    For example, the `signal` for a deterministic signal given by

    $$
    h(t) = a\sin(b t)
    $$

    ``` py
    def signal(toas, a, b):

        return a * numpy.sin(b * toas)
    ```

???+ example "Model file example"
    Here we give examples of model files for both stochastic and deterministic signals.

    === "Stochastic"
        Model file for a stochastic signal with a broken power-law spectrum

        $$
        h^2\Omega_{\rm GW}(f) = A_* \frac{f/f_*}{1+f^2/f_*^2}
        $$

        whose parameters, $A_*$ and $f_*$, are assumed to have a log-uniform prior between
        $[10^{-14},10^{-6}]$ and $[10^{-10},10^{-6}]$ respectively.

        ``` py 
        from ptarcade import parameter

        parameters = {
                    'log_A_star' : parameter.Uniform(-14, -6, True),
                    'log_f_star' : parameter.Uniform(-10, -6, True)
                    }

        def S(x):
            return x / (1 + x**2)

        def spectrum(f, log_A_star, log_f_star):
            A_star = 10**log_A_star
            f_star = 10**log_f_star
            
            return A_star * S(f/f_star)
        ```

    === "Deterministic"
        Model file for a deterministic signal given by 

        $$
        h(t) = A\sin(k t)
        $$

        and where we assumed log-uniform between  $[10^{-14},10^{-6}]$ and $[10^{-10},10^{-6}]$
        for the two model parameters $A$ and $k$ respectively. 

        ``` py
        import numpy
        from ptarcade import parameter


        parameters = {
                    'log_A' : parameter.Uniform(-14, -6, True),
                    'log_k' : parameter.Uniform(-10, -6, True)
                    }

        def signal(toas, log_A, log_k):
            A = 10**log_A
            k= 10**log_k
            
            return A *  numpy.sin(k * toas)
        ```

## Additional settings 
The model file can also contain additional (optional) variables that can be used to control in more details the new-physics signal. Specifically, the following 

<figure markdown>

| Variable name| Variable type | What it is doing                                                                         | Default        |
| :---------:  | :------------:| :------------:                                                                           | :------------:           |
| `name`       | string       | Sets the model name. It used to [name the output directory][out_name].                   | `False`    |
| `smbhb`      | boolean      | If set to `True` the expected signal from SMBHB will be added to the new-physic signal.  | `"np_model"` |
| `goup`       | list         |                                                                                          |         | 

</figure>

!!! info "NG15 model files"
    The model files used in the [NANOGrav 15-year new-physics search][ng15_np] can be found [here][ng15_models].

  [out_name]: ../outputs.md
  [ng15_np]:  link_to_papaer
  [ng15_models]: https://zenodo.org/record/8021439

