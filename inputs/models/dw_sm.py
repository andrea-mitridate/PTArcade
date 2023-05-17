import enterprise.signals.parameter as parameter
import src.models_utils as aux
import numpy as np


name = "domain_walls"  # name of the model

smbhb = False  # set to True if you want to overlay the new-physics signal to the SMBHB signal

parameters = {
    "log10_alpha": parameter.Uniform(-3, 0)("log10_alpha"),
    "log10_T_star": parameter.Uniform(-4, 4)("log10_T_star"),
    "a": parameter.Constant(3)("a"),
    "b": parameter.Uniform(0.5, 1)("b"),
    "c": parameter.Uniform(0.3, 3)("c"),
}

group = ["log10_alpha", "log10_T_star", 'b', 'c']


def S(x, a, b, c):
    """
    | Spectral shape as a function of x=f/f_peak
    """
    return (a + b) ** c / (b * x ** (-a / c) + a * x ** (b / c)) ** c


def spectrum(f, log10_alpha, log10_T_star, a, b, c):
    """
    | Returns the GW energy density as a fraction of the
    | closure density as a function of the parameters of the
    | model:
    |   - f/Hz
    |   - log10(alpha)
    |   - log10(T_star/Gev)
    |   - spectral shape parameters a,b,c
    """

    alpha = 10**log10_alpha
    T_star = 10**log10_T_star

    gs_eq = aux.g_s(aux.T_eq)
    gs_star = aux.g_s(T_star)
    g_star = aux.g_rho(T_star)

    epsilon = 0.7

    f_0 = (
        (gs_eq / gs_star) ** (1 / 3)
        * (np.pi**2 * g_star / 90) ** (1 / 2)
        * aux.T_0
        * T_star
        / aux.M_pl
        * aux.gev_to_hz
    )

    g_facts = g_star * (gs_eq / gs_star) ** (4 / 3)

    return (
        aux.h**2
        * epsilon
        * np.pi
        / 960
        * alpha**2
        * g_facts
        * aux.T_0**4
        / (aux.M_pl**2 * aux.H_0**2)
        * S(f / f_0, a, b, c)
    )
