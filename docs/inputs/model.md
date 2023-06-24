The model file is a simple Python file that, at minimum, needs to contain the following information:

* [Names and prior distributions][priors] of the signal parameters.
* Parametrized form of the signal, which can either be [stochastic][spectrum]
    (and parametrized via its power-spectrum), or [deterministic][signal] (and parametrized
    as a times eries).

In the following, we will explain how these two quantities are defined in the model file. 

  [priors]: #priors
  [spectrum]: #stochastic-signals
  [signal]: #deterministic-signals

## Priors
The priors for the signal parameters are defined via the `parameters` dictionary. The keys of this dictionary must be strings, which will be used as names for the model parameters. The values of this dictionary are [enterprise Parameter][enterprise.signals.parameter.Parameter] objects, the user can create these object via the [`prior`][ptarcade.models_utils.prior] helper function that can be imported from the [ptarcade.models_utils][] module. The first argument that the user needs to pass to the [`prior`][ptarcade.models_utils.prior] function is a string with the name of the prior class they want to use for that parameter, the remaining arguments are used to set the attributes of the prior. By default, parameters are assumed to be common across all pulsars. If the user wants to define a pulsar-dependent parameter, this can be done by passing `common=False` as a keyword argument.

???+ example "Priors Example"
    The `parameters` dictionary of a model described by the parameters $a$ and $b$, which are common among all the pulsars, will look as follows for different choices of the priors:

    === "Uniform Priors"

        ``` py
        parameters = {'a' : prior("Uniform", 0, 1), 'b' : prior("Uniform", 0, 1)} # (1)!
        ```

        1.  In this case, we have chosen uniform priors in the range [0,1] for both
            parameters.

    === "1D Normal Priors"

        ``` py
        parameters = {'a' : prior("Normal", mu=1, sigma=1), 'b' : prior("Normal", 1, 1)} # (1)!
        ```
        
        1.  In this case, we have chosen 1D normal priors with unit mean and variance for
            both parameters.

    === "2D Normal Priors"

        ``` py 
        mu = [1, 1]
        cov = [[1, 0.1],[0.1, 1]]

        parameters = {'a_b' : prior("Normal", mu, cov, size=2)} # (1)!
        ```
        
        1.  In this case, we have chosen a joint 2D normal prior for the model parameters,
        which are grouped in a single two dimensional parameter called `a_b`.

    === "Exponential Priors"

        ``` py
        parameters = {'a' : prior("LinearExp", 1, 1), 'b' : prior("LinearExp", 1, 1)} # (1)!
        ```

        1.  In this case, we have assumed 

??? info "Constructing Priors"

     Notice, how we used both positional and keyword arguments: Both are allowed. These arguments correspond to the functions defined in either [enterprise.signals.parameter][] or [ptarcade.models_utils][]. Below are links to all supported parameters:

     - [enterprise.signals.parameter.Normal][] 
     - [enterprise.signals.parameter.Uniform][] 
     - [enterprise.signals.parameter.TruncNormal][] 
     - [enterprise.signals.parameter.LinearExp][] 
     - [enterprise.signals.parameter.Constant][] 
     - [ptarcade.models_utils.Gamma][].

??? warning "Common Parameters vs. Pulsar-Dependent"

    Parameters are assumed to be common by default. If they are pulsar-dependent, you **must** pass `common=False` as a keyword argument to `prior`. 
    For example, if we want to set the `b` parameters of previous examples to be pulsar-dependent, we can do that as follows 

    ```py
    parameters = {'a' : prior("Uniform", 0, 1), 'b' : prior("Uniform", 0, 1, common=False)} # (1)!
    ```

    1.  In this example, `b` is a pulsar-dependent parameter. By default, the parameters are common to all pulsars in the PTA.

## Stochastic Signals
Stochastic signals are defined via the `spectrum` function. The first parameter of this function should be named `f` and it is supposed to be a [NumPy array][numpy] containing the frequencies (in units of Hz) at which the spectrum will be evaluated. The names of the remaining parameters should match the keys of the `parameters` dictionary. The `spectrum` function should return a [NumPy array][numpy] containing the value of $h^2\Omega_{\mathrm{GW}}$ at each of the frequencies in `f`.

??? example "Stochastic Signal Example"
    The `spectrum` function for a model with 

    $$
    h^2\Omega_{\rm GW}(f) = \frac{a}{1+b/f}
    $$

    is given by

    ``` py
    def spectrum(f, a, b):
            
        return a * 1 / (1 + b/f)
    ```
  
  [numpy]: https://numpy.org/doc/stable/reference/generated/numpy.array.html

## Deterministic Signals { #+model.deterministic }
Deterministic signals are defined via the `signal` function. The first parameter of this function should be named `toas` and it is supposed to be a [NumPy array][numpy] containing the times of arrival (TOAs) (in units of seconds) at which the deterministic signal will be evaluated. The name of the remaining parameters should match the keys of the `parameters` dictionary. The `signal` function should return a [NumPy array][numpy] with the same dimensions as `toas` containing the value of the induced shift for each TOA contained in `toas`.

??? example "Deterministic Signal Example"
    For a deterministic signal,

    $$
    s(t) = a\sin(b t),
    $$
    
    the `signal` is given by
    
    ``` py
    def signal(toas, a, b):

        return a * numpy.sin(b * toas)
    ```

---

???+ example "Model File Example"

    === "Stochastic"
        This is a model file for a stochastic signal with a broken power-law spectrum,

        $$
        h^2\Omega_{\rm GW}(f) = A_* \frac{f/f_*}{1+f^2/f_*^2},
        $$

        whose parameters, $A_*$ and $f_*$, are assumed to have a log-uniform prior between
        $[10^{-14},10^{-6}]$ and $[10^{-10},10^{-6}]$, respectively.

        ``` py 
        from ptarcade.models_utils import prior

        parameters = {
                    'log_A_star' : prior("Uniform", -14, -6),
                    'log_f_star' : prior("Uniform", -10, -6)
                    }

        def S(x):
            return x / (1 + x**2)

        def spectrum(f, log_A_star, log_f_star):
            A_star = 10**log_A_star
            f_star = 10**log_f_star
            
            return A_star * S(f/f_star)
        ```

    === "Deterministic"
        This is a model file for a deterministic signal given by 

        $$
        s(t) = A\sin(k t),
        $$

        and assuming log-uniform priors between $[10^{-14},10^{-6}]$ and $[10^{-10},10^{-6}]$
        for the two model parameters $A$ and $k$, respectively. 

        ``` py
        import numpy
        from ptarcade.models_utils import prior


        parameters = {
                    'log_A' : prior("Uniform", -14, -6),
                    'log_k' : prior("Uniform", -10, -6)
                    }

        def signal(toas, log_A, log_k):
            A = 10**log_A
            k= 10**log_k
            
            return A *  numpy.sin(k * toas)
        ```
!!! tip "Model File Flexibility"

    In defining the `spectrum` or `signal` functions in the model file, you have all the 
    flexibility of a normal Python file. You can, for example, define auxiliary functions,
    import and interpolate tabulated data etc.

## Additional Settings 
The model file can also contain additional (optional) variables that can be used to control the new-physics signal in more detail. Specifically, you can control the following:

[`name`](#+model.name){ #+model.name }

:   :octicons-milestone-24: Default: _`"np_model"`_ – 
    This variable can be assigned to a string to specify the model name. This will be used to
    [name the output directory][out_name].

[`smbhb`](#+model.smbhb){ #+model.smbhb }

:   :octicons-milestone-24: Default: _`False`_ – 
    If set to `True`, the expected signal from SMBHBs will be added to the user-specified signal.

!!! info "NG15 Model Files"
    The model files used in the [NANOGrav 15-year new-physics search][ng15_np] can be found [here][ng15_models].

  [out_name]: ../outputs.md
  [ng15_np]:  link_to_papaer
  [ng15_models]: https://zenodo.org/record/8021439

