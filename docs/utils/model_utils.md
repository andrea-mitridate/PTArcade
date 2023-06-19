# Model utilities

The objects contained in the [models_utils][ptarcade.models_utils]
module can be used to facilitate the creation of model files. 

A detailed discussion of the content of this module can be found in the 
reference section, here we show some of its more useful functionalities

* Several handy constants expressed in natural unit can be found are 
pre-defined in the models utilities module. A complete list can be found 
[here][ptarcade.models_utils].

* The effective number of relativistic degrees of freedom contributing to 
the Universe's energy and entropy density is parametrized in the function 
[`g_rho`][ptarcade.models_utils.g_rho] and 
[`g_s`][ptarcade.models_utils.g_s]. These functions, derived by 
interpolating the tabulated data in , can be evaluated both as a funtion of 
temperature (in GeV), or as a funtion of frequency. In the latter case they 
will return the value of these functions at the time of the cosmological 
evolution when GWs with comoving wavenumber $k=2\pi a_0 f$ re-entered the 
horizon. (1)
{ .annotate }

    1.  Here $a_0$ denotes the value of the cosmological scale factor today. 
In our convention $a_0=1$.

* [`spec_importer`][ptarcade.models_utils.spec_importer], this function allows 
to define the spectrum of a stochastic signal by using tabulated data ...
