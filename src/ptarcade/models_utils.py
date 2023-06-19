"""Module providing physical constants and useful cosmological functions.

All the constants are expressed in GeV, unless differently stated.
The values are taken from the Particle Data Group (PDG).

Attributes
----------
G : np.float64
    Newton constant (GeV$^{-2}$)
M_pl : np.float64
    Reduced Planck mass (GeV)
T_0 : np.float64
    Present day temperature of the Universe (GeV)
z_eq : int
    Redshift of matter-radiation equality
T_eq : np.float64
    Temperature of matter-radiation equality (GeV)
h : float
    Scaling factor for Hubble expansion rate
H_0 : np.float64
    Hubble constant (GeV)
H_0_Hz : np.float64
    Hubble constant (Hz)
omega_v : float
    Dark energy density today (Planck 2018)
omega_m : float
    Matter density today (Planck 2018)
omega_r : float
    Radiation density today (Planck 2018)
A_s : np.float64
    Planck 2018 amplitude of primordial scalar power spectrum
f_cmb : float
    CMB pivot scale (Hz)
gev_to_hz : np.float64
    Conversion from GeV to Hz
g_rho_0 : np.float64
    Relativistic degrees of freedom today
g_s_0 : np.float64
    Entropic relativistic degrees of freedom today
priors_type : typing.Literal["Uniform", "Normal", "TruncNormal", "LinearExp", "Constant", "Gamma"]
    Type for parameter priors.
"""
from __future__ import annotations

from collections import UserDict
from collections.abc import Callable
from functools import cache
from importlib.resources import files
from typing import Any, Literal

import natpy as nat
import numpy as np
import scipy.stats as ss
from enterprise.signals import parameter
from enterprise.signals.parameter import function
from numpy._typing import _ArrayLikeFloat_co as array_like
from numpy.typing import NDArray

from ptarcade import fast_interpolate

nat.set_active_units("HEP")


G : np.float64 = 6.67430 * 10**-11 * nat.convert(nat.m**3 * nat.kg**-1 * nat.s**-2, nat.GeV**-2) # Newton constant (GeV**-2)
M_pl : np.float64 = (8 * np.pi *G)**(-1/2) # reduced plank mass (GeV)
T_0 : np.float64 = 2.7255 * nat.convert(nat.K, nat.GeV) # present day temperature of the Universe (GeV)
z_eq : int = 3402 # redshift of  matter-radiation equality
T_eq : np.float64 = T_0 * (1 + z_eq) # temperature of matter-radiation equality (GeV)
h : float = 0.674 # scaling factor for Hubble expansion rate
H_0 : np.float64 = h * 100 * nat.convert(nat.km * nat.s**-1 * nat.Mpc**-1, nat.GeV) # Hubble constant (GeV)
H_0_Hz : np.float64 = H_0 * nat.convert(nat.GeV, nat.Hz) # Hubble constant (Hz)
omega_v : float = 0.6847 # DE density today Planck 2018
omega_m : float = 0.3153 # matter density today Planck 2018
omega_r : float = 9.2188e-5 # radiation density today Planck 2018
A_s : np.float64 = np.exp(3.044)*10**-10 # Planck 2018 amplitude of primordial scalar power spectrum
f_cmb : float = 7.7314e-17 # CMB pivot scale (Hz)
gev_to_hz : np.float64 = nat.convert(nat.GeV, nat.Hz) # conversion from gev to Hz

# tabulated values for the number of relativistic degrees of
# freedom from reference 1803.01038
gs = np.loadtxt(files('ptarcade.data').joinpath('g_star.dat')) # type: ignore

# type to use for priors-building functions
priors_type = Literal["Uniform", "Normal", "TruncNormal", "LinearExp", "Constant", "Gamma"]


def g_rho(x: array_like, is_freq: bool = False) -> array_like:  # noqa: FBT001, FBT002
    """Return the number of relativistic degrees of freedom as a function of T/GeV or f/Hz.

    Parameters
    ----------
    x : array_like
        The temperature(s) [GeV] or frequency/frequencies [Hz].
    is_freq : bool, optional
        True if `x` is a frequency/frequencies, False if temperature(s).
        Defaults to False.

    Returns
    -------
    dof : array_like
        The relativistic degrees of freedom at `x`.

    """
    if is_freq:
        dof = np.interp(x, gs[:, 1], gs[:, 3])

    else:
        dof = np.interp(x, gs[:, 0], gs[:, 3])

    return dof


def g_s(x: array_like, is_freq: bool = False) -> array_like:  # noqa: FBT001, FBT002
    """Return the number of entropic relativistic degrees of freedom as a function of T/GeV or f/Hz.

    Parameters
    ----------
    x : array_like
        The temperature(s) [GeV] or frequency/frequencies [Hz].
    is_freq : bool, optional
        True if `x` is a frequency/frequencies, False if temperature(s).
        Defaults to False.

    Returns
    -------
    dof : array_like
        The entropic relativistic degrees of freedom at `x`.

    """
    if is_freq:
        dof = np.interp(x, gs[:, 1], gs[:, 2])

    else:
        dof = np.interp(x, gs[:, 0], gs[:, 2])

    return dof


# We cache the following functions so that they only run once and then cache the result
# They are never called with a different argument, so they only have to be computed once
@cache
def __g_s_0(T_0: float) -> np.float64:
    """Calculate the entropic relativistic degrees of freedom today.

    This function is cached because it only needs to be ran once. It is a function instead of a constant so that
    if the `g_star.dat` file changes, this value will update as well.

    Parameters
    ----------
    T_0 : float
        The universe's temperature today [GeV].

    Returns
    -------
    float
        The entropic relativistic degrees of freedom today.

    """
    return g_s(T_0) # type: ignore

@cache
def __g_rho_0(T_0: float) -> np.float64:
    """Calculate the relativistic degrees of freedom today.

    This function is cached because it only needs to be ran once. It is a function instead of a constant so that
    if the `g_star.dat` file changes, this value will update as well.

    Parameters
    ----------
    T_0 : float
        The universe's temperature today [GeV].

    Returns
    -------
    float
        The relativistic degrees of freedom today.

    """
    return g_rho(T_0) # type: ignore

g_rho_0: np.float64 = __g_rho_0(T_0)  # relativistic degrees of freedom today
g_s_0: np.float64 = __g_s_0(T_0)  # entropic relativistic degrees of freedom today

# -----------------------------------------------------------
# Additional priors not included in the standard
# enterprise package.
# -----------------------------------------------------------


def GammaPrior(value: float, a: float, loc: float, scale: float) -> float:
    """Prior function for Gamma parameters.

    Parameters
    ----------
    value : float
        The value of the parameter.
    a : float
        The shape parameter of the Gamma distribution.
    loc : float
        The location parameter of the Gamma distribution.
    scale : float
        The scale parameter of the Gamma distribution.

    Returns
    -------
    float
        The probability density of the Gamma distribution at `value`.

    """
    return ss.gamma.pdf(value, a, loc, scale)

def GammaSampler(a: float, loc: float, scale: float, size: int | None  = None) -> NDArray:
    """Sampling function for Gamma parameters.

    Parameters
    ----------
    a : float
        The shape parameter of the Gamma distribution.
    loc : float
        The location parameter of the Gamma distribution.
    scale : float
        The scale parameter of the Gamma distribution.
    size : int, optional
        The number of samples to draw.

    Returns
    -------
    NDArray
        A NumPy array of size `size` containing samples from the Gamma distribution.

    """
    return ss.gamma.rvs(a, loc, scale, size=size)


def Gamma(a: float, loc: float, scale: float, size: int | None = None):
    """Class factory for Gamma parameters.

    Parameters
    ----------
    a : float
        The shape parameter of the Gamma distribution.
    loc : float
        The location parameter of the Gamma distribution.
    scale : float
        The scale parameter of the Gamma distribution.
    size : int, optional
        The number of samples to draw.

    Returns
    -------
    Gamma : parameter.Parameter
        Child class of [enterprise.signals.parameter.Parameter][] for Gamma distribution

    """
    class Gamma(parameter.Parameter):
        """Child class of enterprise.signals.parameter.Parameter."""

        _size = size
        _prior = parameter.Function(GammaPrior, a=a, loc=loc, scale=scale)
        _sampler = staticmethod(GammaSampler)
        _typename = parameter._argrepr("Gamma", a=a, loc=loc, scale=scale)

    return Gamma


# -----------------------------------------------------------
# Helper functions.
# -----------------------------------------------------------

def omega2cross(omega_hh: Callable[..., NDArray], ceffyl : bool = False) -> Callable[..., NDArray]:
    """Convert GW energy density.

    Converts the GW energy density as a fraction of the closure density into the cross-power spectral density
    as a function of the frequency in Hz. This is intended to be used as a decorator on a function that returns
    the GW energy density as a fraction of the closure density.

    Parameters
    ----------
    omega_hh : Callable[..., NDArray]
        The function that returns the GW energy density as a fraction of the closure density.

    ceffyl: bool
        If set to tru use a version compatible with ceffyl, if set to false a version compatible with 
        ENTERPRISE

    Returns
    -------
    Callable[..., NDArray]
        A function that returns the cross-power spectral density as a function of the frequency in Hz.

    """
    if ceffyl:
        @function
        def cross(f: NDArray, Tspan: float, **kwargs):

            # fraction of the critical density in GWs
            h2_omega = omega_hh(f, **kwargs)

            # characteristic strain spectrum h_c(f)
            hcf = H_0_Hz / h * np.sqrt(3 * h2_omega / 2) / (np.pi * f)

            # cross-power spectral density S(f) (s^3)
            sf = (hcf**2 / (12 * np.pi**2 * f**3)) / Tspan

            return sf

    else:
        @function
        def cross(f: NDArray, components: int = 2, **kwargs):

            df = np.diff(np.concatenate((np.array([0]), f[::components])))

            # fraction of the critical density in GWs
            h2_omega = omega_hh(f, **kwargs)

            # characteristic strain spectrum h_c(f)
            hcf = H_0_Hz / h * np.sqrt(3 * h2_omega / 2) / (np.pi * f)

            # cross-power spectral density S(f) (s^3)
            sf = (hcf**2 / (12 * np.pi**2 * f**3)) * np.repeat(df, components)

            return sf

    return cross


def prep_data(path: str) -> tuple[list[NDArray], NDArray, NDArray]:
    """Shape tabulated data into a form that can be handled by `interpn`.

    Parameters
    ----------
    path : str
        Path to the tabulated data file.

    Returns
    -------
    grids : list[NDArray]
        The grids of the tabulated data.
    omega_grid : NDArray
        The omega grid of the tabulated data.
    par_names : NDArray
        The names of the parameters in the tabulated data.
    """
    par_names = np.loadtxt(path, max_rows=1, dtype='str')
    data = np.loadtxt(path, skiprows=1)

    spec_col = np.where(par_names=='spectrum')
    omega_grid = data.T[spec_col][0]

    data = np.delete(data, spec_col, axis=1)
    par_names = np.delete(par_names, spec_col)

    grids = [np.unique(row) for row in data.T]

    for idx, par in enumerate(par_names):
        omega_grid = omega_grid[data[:, -idx -1].argsort(kind="mergesort")]
        data = data[data[:, -idx -1].argsort(kind="mergesort")]

    grid_size = [len(x) for x in grids]
    omega_grid = omega_grid.reshape(tuple(grid_size))

    return grids, omega_grid, par_names


def spec_importer(path: str) -> Callable[[NDArray, Any],  NDArray]:
    """Import data and create a fast interpolation function.

    Interpolate the GWB power spectrum from tabulated data. Return a function that interpolates
    over frequency.

    Parameters
    ----------
    path : str
        Path to the tabulated data file.

    Returns
    -------
    Callable[[NDArray, P], NDArray]
        A callable object that interpolates the GWB power spectrum at a given frequency `f` and with given
        parameters `kwargs`.
    """
    info, data = fast_interpolate.load_data(path)
    # info is a list of (name, start, step)

    def spectrum(f: NDArray, **kwargs: Any) -> NDArray:

        # Construct right information format for interpolation
        return fast_interpolate.interp([(start, step, f if name == 'f' else kwargs[name])
                                         for (name, start, step) in info],
                                        data)
    return spectrum # type: ignore


def freq_at_temp(T: array_like) -> array_like:
    """Find frequency today as function of temperature when GW was horizon size.

    Calculates the GW frequency [Hz] today as a function of the universe temperature [GeV]
    when the GW was of horizon size.

    Parameters
    ----------
    T : array_like
        The universe temperature [GeV] at the time when the GW was of horizon size.

    Returns
    -------
    NDArray
        The GW frequency [Hz] today that was of horizon size when the universe was at temperature `T` [GeV].
    """
    f_0 = H_0_Hz / (2 * np.pi)

    T_ratio = T_0 / T # type: ignore
    g_ratio = g_rho_0 / g_rho(T) # type: ignore
    gs_ratio = g_s_0 / g_s(T) # type: ignore

    prefactor = f_0 * (gs_ratio) ** (1 / 3) * T_ratio
    sqr_term = np.sqrt(
        omega_v
        + (gs_ratio**-1 * T_ratio**-3 * omega_m)
        + (g_ratio**-1 * T_ratio**-4 * omega_r)
    )

    return prefactor * sqr_term


def temp_at_freq(f: array_like) -> NDArray:
    """Get the temperature [GeV] of the universe when a gravitational wave of a
    certain frequency [Hz] today was of horizon size.

    Parameters
    ----------
    f : array_like
        The frequency in Hz today.

    Returns
    -------
    NDArray
        The temperature [GeV] when the GW at frequency `f` [Hz] was of horizon size.

    """
    return np.interp(f, gs[:, 1], gs[:, 0], left=np.nan, right=np.nan)


class ParamDict(UserDict):
    """UserDict child class that instantiates common or uncommon parameter priors.

    Examples
    --------
    >>> import numpy as np
    >>> from ptarcade.models_utils import ParamDict, prior
    >>> from pprint import pprint
    >>> parameters = ParamDict(
    ...     {
    ...     "log10_A_dm" : prior("Uniform", -9, -4),
    ...     "log10_f_dm" : prior("Uniform", pmin=-10, pmax=-5.5), # kwargs also work
    ...     "gamma_p" : prior("Uniform", 0, 2 * np.pi, common=False),
    ...     "gamma_e" : prior("Uniform", 0, 2 * np.pi),
    ...     "phi_hat_sq_e" : prior("Gamma", a=1, loc=0, scale=1, common=False),
    ...     }
    ... )
    >>> pprint(parameters, sort_dicts=False)
    {'log10_A_dm': log10_A_dm:Uniform(pmin=-9, pmax=-4),
     'log10_f_dm': log10_f_dm:Uniform(pmin=-10, pmax=-5.5),
     'gamma_p': <class 'enterprise.signals.parameter.Uniform.<locals>.Uniform'>,
     'gamma_e': gamma_e:Uniform(pmin=0, pmax=6.283185307179586),
     'phi_hat_sq_e': <class 'ptarcade.models_utils.Gamma.<locals>.Gamma'>}

    """

    def __setitem__(self, key: str, prior: parameter.Parameter):
        if prior.common:
            super().__setitem__(key, prior(key))
        else:
            super().__setitem__(key, prior)


def prior(name: priors_type, *args: Any, **kwargs: Any) -> parameter.Parameter:
    """Wrap enterprise prior creation.

    This function wraps the class factories in [enterprise.signals.parameter][].
    It functions exactly the same as the original, except that it accepts additional
    `kwargs` and sets an additional attribute `common`. This attribute refers to
    whether the parameter this prior corresponds to is common to all pulsars. In the
    original implementation in enterprise, the way that this works is ambiguous. With
    this function, it is explicit. If the user does not pass `common=False` as a `kwarg`,
    then `common` defaults to `True`. This attribute will be used by
    [ptarcade.models_utils.ParamDict][] objects in the model files.

    Parameters
    ----------
    name : priors_type
        The prior to use.
    *args
        Positional arguments passed to the prior factory.
    **kwargs
        kwargs passed to the prior factory.

    Returns
    -------
    prior : parameter.Parameter
        The configured prior.

    Raises
    ------
    ValueError
        If the prior name passed does not exist within [enterprise.signals.parameter]

    """
    # If "common" is passed as kwarg, remove it from the kwarg dictionary and store it.
    # If it wasn't passed, set it to True
    common = kwargs.pop("common", True)

    # Check if the user passed a correct prior name.
    # If they didn't, print an informative message
    try:
        prior_factory = getattr(parameter, name)
    except AttributeError:
        try:
            prior_factory = globals()[name]
        except KeyError:
            err = f"ERROR: the `name` must be one of {priors_type=}, but you entered {name=}."
            raise ValueError(err) from None

    # Use enterprise's class factory
    prior_obj = prior_factory(*args, **kwargs)

    # Store the `common` arg for later use
    prior_obj.common = common

    return prior_obj
