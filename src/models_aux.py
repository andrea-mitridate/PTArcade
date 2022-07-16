import os
from enterprise.signals.parameter import function
import enterprise.signals.parameter as parameter
from scipy import interpolate
import scipy.stats as ss
import numpy as np

cwd = os.getcwd()

# -----------------------------------------------------------
# Physical constants and useful cosmological functions
# -----------------------------------------------------------

# Hubble rate today / h (100 km/s/Mpc)
h0 = 3.24078E-18 # Hz

# tabulated values for the number of relativistic degrees of 
# freedom from reference 1803.01038
gs = np.loadtxt(cwd + '/inputs/models/models_data/g_star.dat')

def g_rho(T):
    """
    | Returns the number of relativistic degrees of 
    | freedom as a function of T/GeV
    """
    return np.interp(T, gs[:,0], gs[:,1])


def g_s(T):
    """
    | Returns the number of entropic relativistic 
    | degrees of freedom as a function of T/GeV
    """
    return np.interp(T, gs[:,0], gs[:,3])


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
        hcf = np.sqrt(3 * h2_omega / 2) * h0 / (np.pi * f)

        # cross-power spectral density S(f) (s^3)s
        sf = (hcf**2 / (12 * np.pi**2 * f**3)) * np.repeat(df, components)

        return sf

    return cross


def prep_data(data):
    """
    Shapes tabulated data in a form that can be handled by interpn.
    """

    n_col = len(data.T)

    grids = [[]] * n_col
    grid_size = []
    for idx, row in enumerate(data.T):
        grids[idx] = np.unique(row)
        grid_size.append(len(np.unique(row)))

    
    grids[-1] = np.reshape(data.T[-1], tuple(grid_size[:-1]))

    return grids


def spec_importer(path):
    """
    Interpolate the GWB power spectrum from tabulated data. 
    """

    f = open(path)
    par_names = f.readline().strip().split('\t')
    f.close()

    data = np.loadtxt(path, skiprows=1)
    data = prep_data(data)

    def spectrum(f, **kwargs):
        points = np.reshape(f, (len(f),-1))

        for name in par_names:
            if name != 'f' and name != 'spectrum':
                x = [[kwargs[name]]] * len(f)
                points = np.hstack((points, x))

        return interpolate.interpn((data[:-1]), data[-1], points)

    return spectrum

def spec_importer_2(path):
    """
    Interpolate the GWB power spectrum from tabulated data. 
    """

    f = open(path)
    par_names = f.readline().strip().split('\t')
    f.close()

    data = np.loadtxt(path, skiprows=1)
    data = prep_data(data)

    def spectrum(f, **kwargs):
        points = np.reshape(f, (len(f),-1))

        for name in par_names:
            if name != 'f' and name != 'spectrum':
                x = [[kwargs[name]]] * len(f)
                points = np.hstack((points, x))

        return interpolate.interpn((data[:-1]), data[-1], points)

    return spectrum


