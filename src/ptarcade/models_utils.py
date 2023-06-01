import os
from functools import cache

import enterprise.signals.parameter as parameter
import natpy as nat
import numpy as np
import scipy.stats as ss
from enterprise.signals.parameter import function
from src import fast_interpolate
from importlib.resources import files


# -----------------------------------------------------------
# Physical constants and useful cosmological functions
# all the constants are expressed in GeV
# unless differently stated, the values are taken from the PDG
# -----------------------------------------------------------
nat.set_active_units("HEP")

G = 6.67430 * 10**-11 * nat.convert(nat.m**3 * nat.kg**-1 * nat.s**-2, nat.GeV**-2) # Newton constant (GeV**-2)
M_pl = (8 * np.pi *G)**(-1/2) # reduced plank mass (GeV)
T_0 = 2.7255 * nat.convert(nat.K, nat.GeV) # present day temperature of the Universe (GeV)
z_eq = 3402 # redshift of  matter-radiation equality
T_eq = T_0 * (1 + z_eq) # temperature of matter-radiation equality (GeV)
h = 0.674 # scaling factor for Hubble expansion rate 
H_0 = h * 100 * nat.convert(nat.km * nat.s**-1 * nat.Mpc**-1, nat.GeV) # Hubble constant (GeV)
H_0_Hz = H_0 * nat.convert(nat.GeV, nat.Hz) # Hubble constant (Hz)
omega_v = 0.6847 # DE density today Planck 2018
omega_m = 0.3153 # matter density today Planck 2018
omega_r = 9.2188e-5 # radiation density today Planck 2018
A_s = np.exp(3.044)*10**-10 # Planck 2018 amplitude of primordial scalar power spectrum
f_cmb = 7.7314e-17 # CMB pivot scale (Hz)
gev_to_hz = nat.convert(nat.GeV, nat.Hz) # conversion from gev to Hz

# tabulated values for the number of relativistic degrees of
# freedom from reference 1803.01038
gs = np.loadtxt(files('ptarcade.data').joinpath('g_star.dat'))


def g_rho(x, is_freq=False):
    """
    | Returns the number of relativistic degrees of 
    | freedom as a function of T/GeV or f/Hz.

    :param x: The temperature(s) [GeV] or frequency/frequencies [Hz]
    :param is_freq: True if `x` is a frequency/frequencies, False if temperature(s)
    :return: the relativistic degrees of freedom at `x`
    """

    if is_freq:
        dof = np.interp(x, gs[:, 1], gs[:, 3])

    else:
        dof = np.interp(x, gs[:, 0], gs[:, 3])

    return dof


def g_s(x, is_freq=False):
    """
    | Returns the number of entropic relativistic degrees of
    | freedom as a function of T/GeV or f/Hz.

    :param x: The temperature(s) [GeV] or frequency/frequencies [Hz]
    :param is_freq: True if `x` is a frequency/frequencies, False if temperature(s)
    :return: the entropic relativistic degrees of freedom at `x`
    """

    if is_freq:
        dof = np.interp(x, gs[:, 1], gs[:, 2])

    else:
        dof = np.interp(x, gs[:, 0], gs[:, 2])

    return dof


# We cache the following functions so that they only run once and then cache the result
# They are never called with a different argument, so they only have to be computed once
@cache
def __g_s_0(T_0):
    """Calculate the entropic relativistic degrees of freedom today.

    This function is cached because it only needs to be ran once. It is a function instead of a constant so that if
    the g_star.dat file changes, this value will update as well.

    :param float T_0: The universe's temperature today [GeV].
    :return: The entropic relativistic degrees of greedom today.
    :rtype: float."""
    return g_s(T_0)


@cache
def __g_rho_0(T_0):
    """Calculate the relativistic degrees of freedom today.

    This function is cached because it only needs to be ran once. It is a function instead of a constant so that if
    the g_star.dat file changes, this value will update as well.

    :param float T_0: The universe's temperature today [GeV].
    :return: The relativistic degrees of greedom today.
    :rtype: float."""

    return g_rho(T_0)


g_rho_0 = __g_rho_0(T_0)  # relativistic degrees of freedom today
g_s_0 = __g_s_0(T_0)  # entropic relativistic degrees of freedom today

# -----------------------------------------------------------
# Additional priors not included in the standard 
# enterprise package. 
# -----------------------------------------------------------


def GammaPrior(value, a, loc, scale):
    """Prior function for Uniform parameters."""
    return ss.gamma.pdf(value, a, loc, scale)


def GammaSampler(a, loc, scale, size=None):
    """Sampling function for Uniform parameters."""
    return ss.gamma.rvs(a, loc, scale, size=size)


def Gamma(a, loc, scale, size=None):

    class Gamma(parameter.Parameter):
        _size = size
        _prior = parameter.Function(GammaPrior, a=a, loc=loc, scale=scale)
        _sampler = staticmethod(GammaSampler)
        _typename = parameter._argrepr("Gamma", a=a, loc=loc, scale=scale)

    return Gamma


# -----------------------------------------------------------
# Helper functions.
# -----------------------------------------------------------

def omega2cross(omega_hh):
    """
    | Converts the GW energy density as a fraction of the 
    | closure density into the cross-power spectral density
    | as a funtion of the frequency in Hz.
    """

    @function
    def cross(f, components=2, **kwargs):

        df = np.diff(np.concatenate((np.array([0]), f[::components])))

        # fraction of the critical density in GWs
        h2_omega = omega_hh(f, **kwargs)

        # characteristic strain spectrum h_c(f)
        hcf = H_0_Hz / h * np.sqrt(3 * h2_omega / 2) / (np.pi * f)

        # cross-power spectral density S(f) (s^3)
        sf = (hcf**2 / (12 * np.pi**2 * f**3)) * np.repeat(df, components)

        return sf

    return cross


def prep_data(path):
    """
    Shapes tabulated data in a form that can be handled by interpn.
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


def spec_importer(path):
    """
    Interpolate the GWB power spectrum from tabulated data. 
    """

    info, data = fast_interpolate.load_data(path)
    # info is a list of (name, start, step)

    def spectrum(f, **kwargs):

        # Construct right information format for interpolation
        return fast_interpolate.interp([(start, step, f if name == 'f' else kwargs[name])
                                         for (name, start, step) in info],
                                        data)
    return spectrum


def freq_at_temp(T):
    """Calculate GW frequency [Hz] today as function of universe temperature [GeV]
    when the GW was of horizon size.

    :param Union[NDArray, float] T: Universe temperature [GeV] at time when GW was of horizon size
    :return: GW of frequency f [Hz] today that was of horizon size when universe was at temperature `T` [GeV]
    :rtype: Union[NDArray, float]
    """

    f_0 = H_0_Hz / (2 * np.pi)

    T_ratio = T_0 / T
    g_ratio = g_rho_0 / g_rho(T)
    gs_ratio = g_s_0 / g_s(T)

    prefactor = f_0 * (gs_ratio) ** (1 / 3) * T_ratio
    sqr_term = np.sqrt(
        omega_v
        + (gs_ratio**-1 * T_ratio**-3 * omega_m)
        + (g_ratio**-1 * T_ratio**-4 * omega_r)
    )

    return prefactor * sqr_term


def temp_at_freq(f):
    """Get the temperature [GeV] of the universe when a gravitational wave of a
    certain frequency [Hz] today was of horizon size.

    :param Union[NDArray, float] f: Frequency in Hz today
    :return: Temperature [GeV] when GW at frequency `f` [Hz] was of horizon size
    :rtype: Union[NDArray, float]
    """

    return np.interp(f, gs[:, 1], gs[:, 0], left=np.nan, right=np.nan)
