import os
from enterprise.signals.parameter import function
import enterprise.signals.parameter as parameter
from scipy import interpolate
import scipy.stats as ss
import numpy as np
import natpy as nat 

cwd = os.getcwd()

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

# tabulated values for the number of relativistic degrees of 
# freedom from reference 1803.01038
gs = np.loadtxt(cwd + '/inputs/models/models_data/g_star.dat')


def g_rho(x, is_freq=False):
    """
    | Returns the number of relativistic degrees of 
    | freedom as a function of T/GeV or f/Hz.

    :param x: The temperature(s) [GeV] or frequency/frequencies [Hz]
    :param is_freq: True if `x` is a frequency/frequencies, False if temperature(s)
    :return: the relativistic degrees of freedom at `x`
    """

    if is_freq:
        dof = np.interp(x, gs[:, -1], gs[:, 1])

    else:
        dof = np.interp(x, gs[:, 0], gs[:, 1])

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
        dof = np.interp(x, gs[:, -1], gs[:, 3])

    else:
        dof = np.interp(x, gs[:, 0], gs[:, 3])

    return dof


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
        hcf = H_0 / h * np.sqrt(3 * h2_omega / 2) / (np.pi * f) * nat.convert(nat.GeV, nat.Hz)

        # cross-power spectral density S(f) (s^3)s
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

    grids, omega_grid, par_names = prep_data(path)

    def spectrum(f, **kwargs):

        points = np.zeros((len(f), len(kwargs) + 1))

        for idx, par in enumerate(par_names):
            if par == 'f':
                points.T[idx] = f

            else:
                points.T[idx] = kwargs[par] * np.ones(len(f))

        return interpolate.interpn((grids), omega_grid, points)

    return spectrum


