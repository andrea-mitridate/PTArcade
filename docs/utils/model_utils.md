# Model Utilities

The objects contained in the [models_utils][ptarcade.models_utils]
module can be used to facilitate the creation of model files.

A detailed discussion of this module can be found in the 
reference section. Here, we highlight some of its most useful functionalities.

* Several handy constants expressed in natural units are 
pre-defined in [models_utils][ptarcade.models_utils]. A complete list can be found 
[here][ptarcade.models_utils].

* The effective number of relativistic degrees of freedom contributing to 
the Universe's energy and entropy density is parametrized in the functions
[`g_rho`][ptarcade.models_utils.g_rho] and 
[`g_s`][ptarcade.models_utils.g_s]. These functions, derived by 
interpolating the tabulated data in #REFERENCE MISSING, can be evaluated both as a function of 
temperature (in GeV), or as a function of frequency (in Hz). In the latter case, they 
will return the value of these functions at the time of the cosmological 
evolution when GWs with comoving wavenumber $k=2\pi a_0 f$ re-entered the 
horizon. (1)
{ .annotate }

    1.  Here $a_0$ denotes the value of the cosmological scale factor today. 
In our convention, $a_0=1$.

* The function [`spec_importer`][ptarcade.models_utils.spec_importer] allows 
to define the spectrum of a stochastic signal by using tabulated data ... #WHAT ARE THE "..." REFERRING TO?
