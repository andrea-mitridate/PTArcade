import enterprise.signals.parameter as parameter
import src.models_utils as aux
import numpy as np

name = 'igw'  # name of the model

smbhb = True  # set to True if you want to overlay the new-physics signal to the SMBHB signal

parameters = {
    'n_t': parameter.Uniform(0, 5)('n_t'),
    'log10_r': parameter.Uniform(-10, 0)('log10_r'),
    'log10_T_rh': parameter.Uniform(-3, 3)('log10_T_rh')  # this in GeV
    }

group = []


def transfer_func(f, f_rh):
    """Calculate the transfer function as a function of GW frequency.

    :param Union[NDArray, float] f: Frequency of GW [Hz] today
    :param Union[NDArray, float] f_rh: Frequency of GW that was of horizon size at reheating
    :return: Transfer function of a GW with frequency f [Hz]
    :rtype: Union[NDArray, float]
    """

    f_ratio = f / f_rh

    return (1 - 0.22 * f_ratio**1.5 + 0.65*f_ratio**2)**-1


def power_spec(f, n_t, r):
    """Calculate primordial tensor power spectrum.

    :param Union[NDArray, float] f: Frequency [Hz] of GW
    :param float n_t: Tensor spectral index
    :param float r: Tensor to scalar ratio
    :return: Primordial tensor power spectrum
    :rtype: Union[NDArray, float]
    """

    return r * aux.A_s * (f / aux.f_cmb)**n_t


@aux.omega2cross
def spectrum(f, n_t, log10_r, log10_T_rh):
    """Calculate GW energy density.

    Returns the GW energy density as a fraction of the closure density as a
    function of the parameters of the model:

    :param Union[NDArray, float] f: Frequency [Hz] of GW today
    :param float n_t: Tensor spectral index
    :param float log10_r: Tensor to scalar ratio in log10 space
    :param float log10_T_rh: Temperature [GeV] at reheating in log10 space
    :return: GW energy density as a fraction of the closure density
    :rtype: Union[NDArray, float]
    """

    r = 10**log10_r
    T_rh = 10**log10_T_rh
    f_rh = aux.freq_at_temp(T_rh)

    # Create a a copy of f. Replace each value where f > f_rh with f_rh
    f_constr = np.where(f <= f_rh, f, f_rh)

    prefactor = (
        (aux.omega_r / 24) * (aux.g_rho(f_constr, is_freq=True) / aux.g_rho_0) *
        (aux.g_s_0 / aux.g_s(f_constr, is_freq=True))**4/3
        )

    return aux.h**2 * prefactor * power_spec(f, n_t, r) * transfer_func(f, f_rh)
