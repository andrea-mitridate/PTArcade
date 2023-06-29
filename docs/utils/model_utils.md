# Model Utilities

The objects contained in the [models_utils][ptarcade.models_utils]
module can be used to facilitate the creation of model files.

A detailed discussion of this module can be found in the 
reference section. Here, we highlight some of its most useful functionalities.

* Several handy constants expressed in natural units are 
pre-defined in [models_utils][ptarcade.models_utils]. A complete list can be found 
[here][ptarcade.models_utils].

* The effective number of relativistic degrees of freedom contributing to 
the Universe's energy and entropy densities is parametrized in the functions
[`g_rho`][ptarcade.models_utils.g_rho] and 
[`g_s`][ptarcade.models_utils.g_s]. These functions, derived by 
interpolating the tabulated data in [Saikawa et al.][gs], can be evaluated both as a functions of 
temperature (in GeV), or as a functions of frequency (in Hz). In the latter case, they 
will return the value of these functions at the time of the cosmological 
evolution when GWs with comoving wavenumber $k=2\pi a_0 f$ re-entered the 
horizon. (1)
{ .annotate }

    1.  Here $a_0$ denotes the value of the cosmological scale factor today. 
In our convention, $a_0=1$.

* The function [`spec_importer`][ptarcade.models_utils.spec_importer] allows the user to define the spectrum of a stochastic signal by using tabulated data. This is useful if the spectrum you are interested in is only evaluated numerically without a closed analytical expression for the stochastic signal amplitude $h^2 \Omega_{\textrm{GW}}$. [`spec_importer`][ptarcade.models_utils.spec_importer] expects the path to an HDF5 file containing the spectrum as a function of frequency and eventual other parameters. It returns a callable function of the frequency $f$ and any other relevant parameters. In the example below, we interpolate a spectrum parametrized by frequency and one additional parameter $p$.

    
    ```py
    import os
    from ptarcade.models_utils import spec_importer

    path = "/This/Is/A/Path/to/the/HDF5/File/spectrum.h5"

    log_spectrum = spec_importer(path)

    def spectrum(f, p):
        return 10**log_spectrum(np.log10(f), p = p)
    ```
In this example, the HDF5 file was generated from a plain-text file with the following formatting:
    ```
    p	 f	        spectrum
    -1	 -10.000000	-19.000000
    -1	  -9.950000	-18.900000
    -1	  -9.900000	-18.800000
    ...
    -0.9 -10.000000	-19.100000
    -0.9  -9.950000	-19.000000
    -0.9  -9.900000	-18.900000
    ...
    ```
    PTArcade provides [`fast_interpolate.reformat`][ptarcade.models_utils.fast_interpolate.reformat] to convert such plain-text files to an HDF5 file that [`fast_interpolate.interp`][ptarcade.models_utils.fast_interpolate.interp] will use to quickly interpolate tabulated data. The plain-text files must meet the following requirements:

    * The file has a header with at least **spectrum** and **f** present
    * Each column is evenly spaced
    * The **f** column must be last if **spectrum** is not. If **spectrum** is last, **f** must be the second-to-last column. 
    
    [`fast_interpolate.reformat`][ptarcade.models_utils.fast_interpolate.reformat] will convert the supplied plain-text file to an HDF5 file at a specified destination with the the following HDF5 datasets:
    
    * `parameter_names` - this dataset contains the parameter names from the header other than **spectrum** 
    * `spectrum` - this dataset contains the **spectrum** data from the original file
    * There will be one additional dataset for each parameter other than **spectrum**. These datasets will contain two values: the minimum value the parameter can take and the step size. The example file above would generate such datasets for *f* and *p*. Assuming the HDF5 file has been read into memory as **data**, then you would have the following:
    ```py
    print(data["p"])
    [-1.0, 0.1]

    print(data["f"]
    [-10.0, 0.05]
    ```

[gs]: https://arxiv.org/abs/2005.03544