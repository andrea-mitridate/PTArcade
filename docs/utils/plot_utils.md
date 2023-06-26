The [`plot_utils`][ptarcade.plot_utils] module contains functions that
can be used to plot the MCMC chains produced by PTArcade. On this page,
we give some examples of the module functionalities. A more detailed
discussion can be found on its [reference page][ptarcade.plot_utils].

[`plot_chains`][ptarcade.plot_utils.plot_chains]

:   This function can be used to produce [trace plots][trace]
    of the chains. For example, let's say that you have a set of chains
    in `./out/model/`. You can produce their [trace plots][trace] as follows:

    ```python
    from ptarcade import chain_utils as c_utils
    from ptarcade import plot_utils as p_utils

    params, chain = c_utils.import_chains('./chains/np_model/')

    p_utils.plot_chains(chain, params)

    ```
This will produce the following:
<figure markdown>
  ![Image title](../assets/images/ex_trace.png){ width="800" }
</figure>

By default, [`plot_chains`][ptarcade.plot_utils.plot_chains] will only produce
trace plots for the user-specified parameters which are common across pulsars,
together with the trace plots for the MCMC parameters (likelihood, posterior, and
hypermodel index). If you want to produce trace plots for a different subset of the 
parameters, you can do it by using the `params_name` argument. 

[`plot_posteriors`][ptarcade.plot_utils.plot_posteriors]

:   This function can be used to produce posterior plots from MCMC chains:

    ```python
    from ptarcade import chain_utils as c_utils
    from ptarcade import plot_utils as p_utils

    params, chain = c_utils.import_chains('./chains/np_model/')

    p_utils.plot_posteriors([chain], [params])

    ```
    This will produce the plot below
<figure markdown>
  ![Image title](../assets/images/ex_post.png){ width="600" }
</figure>


    



[trace]: https://www.statlect.com/fundamentals-of-statistics/Markov-Chain-Monte-Carlo-diagnostics#hid9
