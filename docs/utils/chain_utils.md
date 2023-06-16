The [`chains_utils`][ptarcade.chains_utils] module contains several functions 
that can be used to analyze the MCMC chains produced by PTAracade. In this page
we just highlight some of its functionalities, a more detailed discussion of 
this module can be found in its [reference page][ptarcade.chains_utils].

[`import_chains`][ptarcade.chains_utils.import_chains]

:   This function can be used to load chains and model parameters of a PTArcade 
    run. Just type 

    ```python
    import ptarcade.chains_utils as utils

    params, chain = utils.import_chains('path_to_chains_folder')
    ```

    The [`import_chains`][ptarcade.chains_utils.import_chains] function is 
    particularly useful when you have multiple chains for the same model that 
    you want to merge. In this case [`import_chains`][ptarcade.chains_utils.import_chains]
    will do that for you by merging all the chains that are located inside
    the path that you pass to it. By default, [`import_chains`][ptarcade.chains_utils.import_chains]
    will also remove the first 25% of each chain before merging it, if you
    want to change the amount of burn-in you can do via the `burn_frac` argument. 

    Finally, notice that by default [`import_chains`][ptarcade.chains_utils.import_chains]
    will only load the part of the chains corresponding to user specified parameters, likelihood,
    posterior, and hypermodel index. If you want to also load red noise and eventual DM
    parameters you can do that by setting the `quick_import = False`. 
    


[`compute_bf`][ptarcade.chains_utils.compute_bf]

:   This function can be used to compute Bayes factors from runs where [`mod_sel = True`][mod_sel]
    in the configuration file. You can do this as follows

    ```python
    import ptarcade.chains_utils as utils

    params, chain = utils.import_chains('path_to_chains_folder')

    bf, bf_err = utils.compute_bf(chain, params)
    ```

    This will give an estimate for the Bayes factor for the comparison of the
    user specified signal against the SMBHB signal, and the associated error. 
    By default, the Bayes factor is calculated by dividing the number of points
    in the chain that fall in the hypermodel bin of the user specified signal by
    the numbder of points falling in the bin of the reference SMBHB model. For a 
    more precise estimate of the Bayes factor and associated error, you can set
    `bootstrap=True`. In this case the Bayes factor and its standard deviation
    will be derived by using bootsrapping methods.

[out]: ../outputs.md
[import]: ptarcade.chains_utils.import_chains
[mod_sel]: ../inputs/config.md#+config.mod_sel