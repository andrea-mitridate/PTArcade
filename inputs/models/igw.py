import enterprise.signals.parameter as parameter
import src.models_utils as aux
import numpy as np
import scipy.constants as scon
import natpy as nat
import numpy.typing as npt

name = 'igw' # name of the model

smbhb = False # set to True if you want to overlay the new-physics signal to the SMBHB signal

parameters = {
    'n_t':parameter.Uniform(-2,2)('n_t'),
    'log10_r':parameter.Uniform(-7,0)('log10_r'),
    'log10_T_rh':parameter.Uniform(-6,16)('log10_T_rh') # this in GeV
    }

group = []


def transfer_func(f: float, f_rh: float) -> float:
    """Calculate the transfer function as a function of GW frequency."""


    f_ratio = f / f_rh

    return (1 - 0.22 * f_ratio**1.5 + 0.65**f_ratio**2)**-1

def power_spec(f: float, n_t: float, r: float) -> float:
    """Calculate primordial tensor power spectrum."""

    return r * aux.A_s * ( f / aux.f_cmb )**n_t


@aux.omega2cross
def spectrum(f: npt.NDArray, n_t: float, log10_r: float, log10_T_rh: float) -> npt.NDArray:
    """Calculate GW energy density.

    Returns the GW energy density as a fraction of the closure density as a
    function of the parameters of the model:

    :param float f: Frequency in Hz.
    :param float n_t: Tensor spectral index.
    :param float log10_r: Tensor to scalar ratio in log10 space.
    :param float log10_T_rh: Temperature [GeV] at reheating in log10 space.
    :returns: GW energy density as a fraction of the closure density.
    :rtype: float
    """

    r = 10**log10_r
    T_rh = 10**log10_T_rh
    f_rh = aux.freq_at_temp(T_rh)


    idx = f <= f_rh # this creates an array of booleans
    # use this if f<=f_rh
    prefactor_lt = (
        ( aux.omega_r / 24 ) * ( aux.g_rho(aux.temp_at_freq(f)) / aux.g_rho_0 ) *
        ( aux.g_s_0 / aux.g_s(aux.temp_at_freq(f)) )**4/3
        )

    # use this if f>f_rh
    prefactor_gt = (
        ( aux.omega_r / 24 ) * ( aux.g_rho(aux.temp_at_freq(f_rh)) / aux.g_rho_0 ) *
        ( aux.g_s_0 / aux.g_s(aux.temp_at_freq(f_rh)) )**4/3
        )

    # Now, f is actually an array so we need an array of prefactors that differ
    # based on if f<=f_rh or f > f_rh
    prefactor = np.ones_like(f)
    prefactor[idx] = prefactor_lt
    prefactor[~idx] = prefactor_gt

    return aux.h**2 * prefactor * power_spec(f, n_t, r) * transfer_func(f, f_rh)
