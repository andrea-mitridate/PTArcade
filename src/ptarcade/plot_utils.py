"""Utilities for plotting.

Attributes
----------
colors : list[str]
    list of colors used in plotting
plt_params : dict[str, Any]
    Plotting parameters to use in plotting functions

"""
from __future__ import annotations

import logging
import math
import warnings
from dataclasses import dataclass
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from getdist import plots
from getdist.mcsamples import MCSamples
from numpy._typing import _ArrayLikeFloat_co as array_like
from numpy.typing import NDArray
from scipy.stats import norm

import ptarcade.chains_utils as utils

log = logging.getLogger("rich")


@dataclass(frozen=True)
class bcolors:
    """Class to hold ANSI escape sequences.

    Attributes
    ----------
    WARNING : str
    FAIL : str
    ENDC : str

    """

    WARNING: str = "\033[0;33m"
    FAIL: str = '\033[0;31m'
    ENDC: str = '\033[0m'

colors = [
        '#E03424',
        '#006FED',
        'gray',
        '#009966',
        '#000866',
        '#336600',
        '#006633',
        'm',
        'r']


plt_params = {
        'axes.linewidth' : 0.5,
        'text.usetex'  : True,
        }


def set_size(width: float,
             fraction: int = 1,
             ratio: float | None = None,
             subplots: tuple[int, int] = (1, 1)) -> tuple[float, float]:
    """Set figure dimensions to avoid scaling in LaTeX.

    Parameters
    ----------
    width : float
        Document textwidth or columnwidth in pts
    fraction : float, optional
        Fraction of the width which you wish the figure to occupy, by default 1
    ratio : float, optional
        Ratio of the height to width of the figure. If None, the golden ratio (sqrt(5)-1)/2 is used,
        by default None
    subplots : tuple[int], optional
        The number of rows and columns of subplots in the figure, by default (1, 1)

    Returns
    -------
    fig_dim : tuple
        Dimensions of figure in inches. The dimensions are calculated based on the parameters, and
        the height is further adjusted based on the number of subplots.

    """
    fig_width_pt = width * fraction

    inches_per_pt = 1 / 72.27

    fig_width_in = fig_width_pt * inches_per_pt
    if not ratio:
        ratio = (5**0.5 - 1) / 2

    fig_height_in = fig_width_in * ratio * (subplots[0] / subplots[1])

    fig_dim = (fig_width_in, fig_height_in)

    return fig_dim


def set_custom_tick_options(
    ax: plt.axis,
    left: bool = True,
    right: bool = True,
    bottom: bool = True,
    top: bool = True,
    label_size: int = 8,
    width: float = 0.5,
    length: float = 3.5,
):
    """
    Sets custom tick options for a matplotlib axis.

    Parameters
    ----------
    ax : matplotlib.pyplot.axis
        The axis to which the tick options will be applied.
    left : bool, optional
        If True, left ticks will be visible. Default is True.
    right : bool, optional
        If True, right ticks will be visible. Default is True.
    bottom : bool, optional
        If True, bottom ticks will be visible. Default is True.
    top : bool, optional
        If True, top ticks will be visible. Default is True.
    label_size : int, optional
        Controls the font size for ticks labels. Default is 8.
    width : float, optional
        Sets the width of the ticks. Default is 0.5.
    length : float, optional
        Sets the length of the ticks. Default is 3.5.

    Returns
    -------
    None

    """

    ax.minorticks_on()

    ax.tick_params(
        which="major",
        direction="in",
        length=length,
        width=width,
        bottom=bottom,
        top=top,
        left=left,
        right=right,
        labelsize=label_size,
        pad=2,
    )
    ax.tick_params(
        which="minor",
        direction="in",
        length=length / 2,
        width=width,
        bottom=bottom,
        top=top,
        left=left,
        right=right,
        labelsize=label_size,
        pad=2,
    )

    return


def plot_chains(
    chain: NDArray,
    params: list[str],
    params_name: dict[str, str] = None,
    label_size: int = 13,
    save: bool = False,
    model_name: str | None = None,
) -> None:
    """
    Plot the MCMC (Markov chain Monte Carlo) chain for the parameters specified by params.

    Parameters
    ----------
    chain : numpy.ndarray
        A numpy array containing the MCMC chain to be plotted.
        Each row corresponds to a step in the chain, and each column corresponds to a parameter.
    params : list of str
        A list with the names of the parameters appearing in the chain.
    params_name : dict, optional
        A dictionary with keys being the names of the parameters in the params list to be plotted,
        and values being the formatted parameters name to be shown in the plots.
        Default is None, which plots all the common parameters + the MCMC parameters without any formatting.
    label_size : int, optional
        Controls the font size for the axis and ticks labels. Default is 13.
    save : bool, optional
        If set to True, the plot is saved in the folder "./plots/". Default is False.
    model_name : str, optional
        A string with the model name. Used to name the output files. Default is None.

    Returns
    -------
    None

    Raises
    ------
    Warning
        If some of the requested parameters do not appear in the parameter list.

    """

    plt.rcParams.update(plt_params)

    params_dic = {}

    if params_name:
        for idx, par in enumerate(params):
            if par in params_name.keys():
                formatted_name = params_name[par]
                params_dic[formatted_name] = idx

        if len(params_dic) != len(params_name):
            log.warning(
                "Some of the requested parameters does not appear in the parameter list"
                f"You suppplied {params_dic=} but {params_name=}"
            )

    else:
        for idx, par in enumerate(params):
            if "+" not in par and "-" not in par and "n_parall" not in par:
                key = par.replace("_", "\_")
                key = rf"$\mathrm{{{key}}}$"
                params_dic[key] = idx

    n_par = len(params_dic)
    n_row = int(n_par**0.5)
    n_col = int(math.ceil(n_par / float(n_row)))

    fig, axs = plt.subplots(n_row, n_col, figsize=(5 * n_col, 2.5 * n_row), sharex=True)

    if n_par == 1:
        axs = np.array([axs])

    for idx, (ax, key) in enumerate(zip(axs.reshape(-1), params_dic)):
        ax.plot(chain[:, params_dic[key]])

        set_custom_tick_options(ax, label_size=label_size)

        ax.set_ylabel(f"{key}", fontsize=label_size)

    for ax in axs[-1]:
        ax.set_xlabel("$\mathrm{MCMC\;sample}$", fontsize=label_size)

    plt.tight_layout()

    if save:
        if model_name:
            plt.savefig(f"./plots/{model_name}_chains.pdf")
        else:
            print("Please specify a model name to save the chain plot.")

    plt.show()

    return


def create_ax_labels(par_names: list[str], labelsize: int = 8) -> None:
    """Format the axes for posterior plots.

    This function fetches the current figure and all its axes, sets custom tick options
    and labels for all axes according to the names of the parameters.

    Parameters
    ----------
    par_names : list[str]
        A list of names of the parameters for each of the axes.
    labelsize : int, optional
        The font size of the labels. Default is 8. If there is only one parameter,
        this value is set to 12.

    Returns
    -------
    None

    Notes
    -----
    - The function assumes that the number of axes in the current figure matches the
      number of parameter names provided. If this is not the case, the behavior is undefined.
    - A label with the name 'A' and alpha=0 is set for the y-axis in the case of a single parameter
      to prevent padding issues.

    """
    # small_font = 5

    N_params = len(par_names)

    f = plt.gcf()
    axs = f.get_axes()

    if N_params == 1:
        labelsize = 12

        set_custom_tick_options(axs[0], width=0.5, length=5, label_size=labelsize)
        axs[0].set_xlabel(par_names[0], fontsize=labelsize, labelpad=10)
        axs[0].set_ylabel("A", alpha=0)  # here just ot prevent padding issues

        return

    # do this loop using combinations_with_replacement from itertools
    for idx in range(N_params):
        for idy in range(N_params - idx):
            id = int(N_params * idx + idy - max(idx * (idx - 1) / 2, 0))
            set_custom_tick_options(axs[id], width=0.5, label_size=labelsize)

            if idx == 0 and idy == 0:
                set_custom_tick_options(axs[id], width=0.5, label_size=labelsize)
                axs[id].set_xlabel(par_names[idx], fontsize=labelsize, labelpad=7)
                axs[id].set_ylabel(par_names[N_params - idy - 1], fontsize=labelsize)
            elif idx == N_params - 1:
                set_custom_tick_options(axs[id], left=False, right=False, width=0.5, label_size=labelsize)
                axs[id].set_xlabel(par_names[idx], fontsize=labelsize, labelpad=7)
            elif idy == N_params - idx - 1:
                set_custom_tick_options(axs[id], left=False, right=False, width=0.5, label_size=labelsize)
            elif idx == 0:
                set_custom_tick_options(axs[id], width=0.5, label_size=labelsize)
                axs[id].set_ylabel(par_names[N_params - idy - 1], fontsize=labelsize)
            elif idy == 0:
                set_custom_tick_options(axs[id], width=0.5, label_size=labelsize)
                axs[id].set_xlabel(par_names[idx], fontsize=labelsize, labelpad=7)
            else:
                set_custom_tick_options(axs[id], width=0.5, label_size=labelsize)

    return


def level_to_sigma(level: float) -> float:
    """Convert confidence level to standard deviation (sigma).

    This function uses the inverse of the cumulative distribution function (CDF) for a
    normal distribution to convert a confidence level to the equivalent number of
    standard deviations (sigma).

    Parameters
    ----------
    level : float
        The confidence level to convert. This should be a fraction between 0 and 1.

    Returns
    -------
    float
        The number of standard deviations corresponding to the provided confidence level.

    Raises
    ------
    SystemExit
        If the provided level is not between 0 and 1.

    """
    if 0 < level < 1:
        return np.sqrt(-2 * np.log(1-level))
    else:
        error = ("The level value needs to be between 0 and 1.")

        log.error(error)
        raise SystemExit


def plot_bhb_prior(plot: matplotlib.figure.Figure, bhb_prior: str, levels: list[float]) -> None:
    """Plot the prior distribution for SMBHB signal.

    Parameters
    ----------
    plot : matplotlib.figure.Figure
        The plot object to which the priors will be added.
        This object should have methods `get_axes_for_params(param1, param2)` that return the Axes
        object for the given parameters.
    bhb_prior : str
        The specific BHB prior to be used, choices are "NG15" and "IPTA2".
    levels : list of float
        A list of threshold levels. The sigma equivalent of these levels will be plotted.

    Returns
    -------
    None

    """
    sigmas = [level_to_sigma(level) for level in levels]

    if bhb_prior == "NG15":
        mu = np.array([-15.61492963, 4.70709637])
        cov = np.array([[0.27871359, -0.00263617], [-0.00263617, 0.12415383]])
    elif bhb_prior == "IPTA2":
        mu = np.array([-15.02928454, 4.14290127])
        cov = np.array([[0.06869369, 0.00017051], [0.00017051, 0.04681747]])

    A_0, gamma_0 = mu

    eig = np.linalg.eig(np.linalg.inv(cov))
    a, b = eig[0]
    R_rot = eig[1]

    t = np.linspace(0, 2 * np.pi, 100)
    Ell = -np.array([(a) ** (-1 / 2) * np.cos(t), (b) ** (-1 / 2) * np.sin(t)])

    Ell_rot = np.zeros((len(sigmas), 2, Ell.shape[1]))
    for idx, sigma in enumerate(sigmas):
        Ell_rot[idx, :, :] = np.dot(R_rot, sigma * Ell)

    lw = 0.5

    ax_0 = plot.get_axes_for_params("gw-bhb-0")
    ax_1 = plot.get_axes_for_params("gw-bhb-1")

    if ax_0 and ax_1:
        A_pts = np.arange(-17, -13, 0.001)
        g_pts = np.arange(2, 6, 0.001)
        ax_0.plot(A_pts, norm.pdf(A_pts, mu[0], cov[0, 0] ** (1 / 2)), color="black", linestyle="dashed", linewidth=lw)

        ax_1.plot(g_pts, norm.pdf(g_pts, mu[1], cov[1, 1] ** (1 / 2)), color="black", linestyle="dashed", linewidth=lw)

    for idx in range(len(sigmas)):
        ax = plot.get_axes_for_params("gw-bhb-0", "gw-bhb-1")

        if ax:
            ax.plot(
                A_0 + Ell_rot[idx, 0, :],
                gamma_0 + Ell_rot[idx, 1, :],
                color="black",
                linestyle="dashed",
                linewidth=lw,
                alpha=0.9,
            )

        ax = plot.get_axes_for_params("gw-bhb-1", "gw-bhb-0")

        if ax:
            ax.plot(
                gamma_0 + Ell_rot[idx, 1, :],
                A_0 + Ell_rot[idx, 0, :],
                "black",
                linestyle="dashed",
                linewidth=lw,
                alpha=0.9,
            )
    return


def corner_plot_settings(
    levels: list[float] | None,
    samples: list[MCSamples],
    one_column: bool,
    legend_size: int = 12,
    fig_width_pt: float | None = None,
) -> plots.GetDistPlotSettings:
    """Configure the settings for corner plots.

    Parameters
    ----------
    levels : list[float], optional
        The list of confidence levels for which the HPI should be computed.
        Each value in the list should be between 0 and 1. Default is None.
    samples : list[MCSamples]
        A list of instances of the MCSamples class, each containing
        multivariate Monte Carlo samples on which the function is operating.
    one_column : bool
        Whether to set the figure width for a one-column format. If False,
        a wider format is used.
    legend_size : int, optional
        Sets the font size of the legend captions.
        Default is 12.
    fig_width_pt : float, optional
        If provided, the figure width in points. This parameter can be used
        to override the default figure widths for one- or two-column formats.
        Default is None.

    Returns
    -------
    plots.GetDistPlotSettings
        An instance of the GetDistPlotSettings class, with the corner plot
        settings configured as specified.

    """

    sets = plots.GetDistPlotSettings()
    sets.prob_y_ticks = False
    sets.figure_legend_loc = "upper right"
    sets.norm_1d_density = "integral"
    sets.alpha_filled_add = 0.8
    sets.legend_fontsize = legend_size
    sets.linewidth = 2.35
    if fig_width_pt:
        sets.fig_width_inch = set_size(fig_width_pt, ratio=1)[0]
    elif one_column:
        sets.fig_width_inch = set_size(246, ratio=1)[0]
    else:
        sets.fig_width_inch = set_size(510, ratio=1, fraction=0.65)[0]

    if levels is not None:
        sets.num_plot_contours = len(levels)
        for sample in samples:
            sample.updateSettings({"contours": levels})

    return sets



def oned_plot_settings(legend_size: int = 9) -> plots.GetDistPlotSettings:
    """Configure the settings for one-dimensional (1D) plots.

    Parameters
    ----------
    legend_size : int, optional
        The size for the legend font.

    Returns
    -------
    plots.GetDistPlotSettings
        An instance of the GetDistPlotSettings class, with the 1D plot
        settings configured as specified.

    """

    sets = plots.GetDistPlotSettings()

    sets.legend_fontsize = legend_size
    sets.prob_y_ticks = False
    sets.linewidth = 1.1
    sets.figure_legend_loc = "upper right"
    sets.norm_1d_density = "integral"
    sets.line_styles = colors

    return sets


def plot_k_bounds(
        plot: plots.GetDistPlotter.triangle_plot,
        samples: list[MCSamples],
        k_levels: list[NDArray],
) -> None:
    """Add the k-ratio bounds to an existing plot.

    Parameters
    ----------
    plot : plots.GetDistPlotter.triangle_plot
        The plot to which the k-ratio bounds should be added.
    samples : list[MCSamples]
        A list of instances of the MCSamples class, each containing
        multivariate Monte Carlo samples on which the function is operating.
    k_levels : list[array_like]
        A list of 1D or 2D arrays representing the K-ratio bounds for
        each sample. Each array should contain the parameter names and
        corresponding K-ratio bound.

    Returns
    -------
    None

    """
    for idx, sample in enumerate(samples):
        if k_levels[idx]:
            for x in k_levels[idx][0]:
                if x[-1]:
                    par = x[0]
                    p0 = float(x[-1])
                    if len(k_levels[idx][0]) == 1:
                        plt.plot(
                            [p0, p0], [0, 10], color=colors[idx], alpha=0.8, lw=0.8)
                    else:
                        ax = plot.get_axes_for_params(par)
                        ax.plot([p0, p0], [0, 10], color=colors[idx], alpha=0.8, lw=0.8)

            for x in k_levels[idx][1]:
                par_1 = x[0]
                par_2 = x[1]
                level = float(x[-1])
                par_1, par_2 = plot.get_param_array(sample, [par_1, par_2])
                ax = plot.get_axes_for_params(par_1, par_2)

                density = plot.sample_analyser.get_density_grid(
                    sample,
                    par_1,
                    par_2,
                    conts=plot.settings.num_plot_contours,
                    likes=plot.settings.shade_meanlikes)

                levels = sorted(np.append([density.P.max() + 1], level))
                cs = ax.contourf(
                    density.x,
                    density.y,
                    density.P,
                    levels,
                    colors="#ffffff00",
                    alpha=0.14,
                    extend="both")

                cs.cmap.set_over("#ffffff00")
                cs.cmap.set_under("gray")
                cs.changed()

                levels = sorted(np.append([density.P.max() + 1], level))
                ax.contour(
                    density.x,
                    density.y,
                    density.P,
                    levels=[level],
                    colors="grey",
                    linewidths=0.5,
                    alpha=1)

    return


def plot_hpi(
        plot: plots.GetDistPlotter.triangle_plot,
        samples: list[MCSamples],
        hpi_points: list[array_like],
) -> None:
    """Add the highest posterior interval (HPI) to an existing plot.

    Parameters
    ----------
    plot : plots.GetDistPlotter.triangle_plot
        The plot to which the HPI should be added.
    samples : list[MCSamples]
        A list of instances of the MCSamples class, each containing
        multivariate Monte Carlo samples on which the function is operating.
    hpi_points : list[array_like]
        A list of arrays representing the HPI for each sample. Each array
        should contain the parameter name and corresponding HPI.

    Returns
    -------
    None

    """
    lw=0.6

    if len(samples) == 1:
        c_adjust = 1
    else:
        c_adjust = 0

    for idx, sample in enumerate(samples):
        hpi = hpi_points[idx]

        for x in hpi:
            for level in x[1]:
                par = x[0]
                x1 = level[0]
                x2 = level[1]
                density = MCSamples.get1DDensity(sample, par)

                if len(hpi) == 1:
                    f = plt.gcf()
                    ax = f.get_axes()[0]
                else:
                    ax = plot.get_axes_for_params(par)

                if x1:
                    p1 = density(x1)[0]
                    ax.plot([x1, x1], [0, p1], ls = 'dashed', color = colors[idx+c_adjust], lw=lw)
                if x2:
                    p2 = density(x2)[0]
                    ax.plot([x2, x2], [0, p2], ls = 'dashed', color = colors[idx+c_adjust], lw=lw)

    return


def print_stats(
        k_levels: list[array_like],
        hpi_points: list[array_like],
        bayes_est: list[dict],
        max_pos:list[dict],
        levels:list[float],
) -> None:
    """Print the statistical summary for each sample.

    Parameters
    ----------
    k_levels : list[array_like]
        A list of arrays representing the K-ratio bounds for each sample. Each array
        should contain the parameter name and corresponding K-ratio bound.
    hpi_points : list[array_like]
        A list of arrays representing the highest posterior interval (HPI) for each sample.
        Each array should contain the parameter name and corresponding HPI.
    bayes_est : list[dict]
        A list of dictionaries, each containing the Bayes estimator for each parameter in the sample.
    max_pos : list[dict]
        A list of dictionaries, each containing the maximum posterior value for each parameter in the sample.
    levels : list[float]
        The list of confidence levels for which the HPI should be computed. Each value
        in the list should be between 0 and 1.

    Returns
    -------
    None

    """
    for idx in np.arange(len(k_levels)):
        print(f'----- Stats for sample #{idx}-----')

        if k_levels[idx] and any(x != [] for x in k_levels[idx]):
            k_level = k_levels[idx][0]

            for par, level in k_level:
                if level:
                    print(f'k-ratio limit = is reached for {par} = {level}')
                else:
                    print(f'k-ratio limit is not reached for {par}')

        else:
            print(f'No 1D k-bounds available for this sample\n')

        hpi = hpi_points[idx]

        for x in hpi:
            for idy, level in enumerate(x[1]):
                par = x[0]
                x1 = level[0]
                x2 = level[1]

                if x1:
                    print(f'Lower {100*levels[idy]}%-HPDI limit is reached for {par} = {10**x1}')
                else:
                    print(f'Lower {100*levels[idy]}%-HPDI limit is reached for {par} does not exist')
                if x2:
                    print(f'Upper {100*levels[idy]}%-HPDI limit is reached for {par} = {10**x2}')
                else:
                    print(f'Upper {100*levels[idy]}%-HPDI limit is reached for {par} does not exist')

        for par, val in bayes_est[idx].items():
            print(f'The Bayes estimator for {par} is {val[0]} +- {val[1]}')

        for par, val in max_pos[idx].items():
            print(f'The maximum posterior value for {par} is {val}')

        print('\n')

    return


def plot_posteriors(
    chains: list[array_like],
    params: list[list[str]],
    par_to_plot: list[list[str]] | None =None,
    par_names: list[list[str]] | None  =None,
    model_id: list[int] | None =None,
    samples_name: list[str] | None =None,
    hpi_levels: list[float]=[0.68, 0.95],
    k_ratio: list[float] | None =None,
    bf: list[float] | None =None,
    ranges: dict[str, array_like] ={},
    levels: list[float] | None =None,
    bhb_prior: str | bool =False,
    one_column: bool=False,
    fig_width_pt: float| None =None,
    labelsize: float=8,
    legend_size: float=12,
    verbose: bool=False,
    save:bool=False,
    plots_dir: Path | str = './plots',
    model_name: str | None=None) -> tuple[matplotlib.figure.Figure, matplotlib.axes.Axes]:
    """
    Plot posterior distributions for the chosen parameters in the chains.

    Parameters
    ----------
    chains : list[array_like]
        list of array_like's. Contains all the chains to be plotted.
    params : list[list[str]]
        list of lists[str]. Contains the name of the parameters appearing in the chains.
    par_to_plot : list[list[str]], optional
        list of lists[str]. Contains the parameter to plot from each chain.
    par_names : lists[list[str]], optional
        Contains the LaTeX formatted names for the plotted parameters of each model.
        If not specified the name in the par files will be used.
    model_id : list[int], optional
        Assuming that the data are generated using hypermodels, specifies the model for
        which to plot the posteriors (default is None and in this case nmodel is taken to be 0).
    samples_name : list[str], optional
        Contains the name of the models associated to each chain, and it's used to
        create the legend. If not specified, the labels in the legend will be Sample 1, Sample 2, etc.
    hpi_levels : list[float], optional
        The list of confidence levels for which the highest posterior interval (HPI)
        should be computed. Each value in the list should be between 0 and 1.
    k_ratio : list[float], optional
        The list of K-ratio bounds for each chain. Each value in the list should be between 0 and 1.
    bf : list[float], optional
        Bayes factor for each chain, used to calculate K-ratio bounds.
    ranges : dict, optional
        The parameter ranges to display on the plot. Keys should be parameter names, and values should be tuples
        or lists specifying the lower and upper bounds of the range.
    levels : list[float], optional
        The list of confidence levels for which the contour should be computed. Each value
        in the list should be between 0 and 1.
    bhb_prior : str, optional
        String indicating the bhb prior to plot (possible choiches are NG15 and IPTA2.
    one_column : bool, optional
        Whether to display the plot in one column format.
    fig_width_pt : float, optional
        The width of the figure in points.
    labelsize : float, optional
        The fontsize for the labels on the plot.
    labelsize : float, optional
        The fontsize for the labels on the plot.
    verbose : bool, optional
        If set to True, the function will print statistical summaries of the data.
    save : bool, optional
        If set to True, the function will save the plot to a PDF file.
    plots_dir : Path | str, optional
        The directory to save plots in. The directory will be created if needed.
    model_name : str, optional
        Model name used to name the output files. If not specified the plot will be
        saved as 'corner.pdf'.

    Returns
    -------
    fig : matplotlib.figure.Figure
        The figure containing the plot.
    axs : list[matplotlib.axes.Axes]
        The axes of the figure.

    """
    plt.rcParams.update(plt_params)

    N_chains = len(chains)

    priors = params
    params = [np.array(list(par.keys())) for par in params]

    if par_to_plot is None:
        par_to_plot = [None] * N_chains
    if model_id is None:
        model_id = [None] * N_chains
    if k_ratio is None:
        k_ratio = [None] * N_chains
    if bf is None:
        bf = [
            utils.compute_bf(chain, params[idx])[0] for idx, chain in enumerate(chains) if k_ratio[idx]]
    if not samples_name:
        samples_name = [f"Sample {idx+1}" for idx in range(N_chains)]

    samples = []
    par_union = []
    k_levels = [[]] * N_chains
    hpi_points = [[]] * N_chains
    bayes_est = [[]] * N_chains
    max_pos = [[]] * N_chains
    for idx, chain in enumerate(chains):
        filtered = utils.chain_filter(
            chain, params[idx], model_id[idx], par_to_plot[idx]
        )
        filtered_priors = {
            k.replace("_", "-"): v
            for k, v in priors[idx].items()
            if k.replace("_", "-") in filtered[1] and (v is not None and not np.isnan(v).any())
        }

        samples.append(
            MCSamples(
                samples=filtered[0],
                names=filtered[1],
                ranges=filtered_priors,
                ignore_rows=1,
            )
        )

        if k_ratio[idx] and bf[idx]:
            k_levels[idx] = utils.get_k_levels(
                sample=samples[-1],
                pars=filtered[1],
                priors=filtered_priors,
                bf=bf[idx],
                k_ratio=k_ratio[idx])
        else:
            k_levels[idx] = None

        if hpi_levels:
            hpi_points[idx] = utils.get_c_levels(
                sample=samples[-1], pars=filtered[1], levels=hpi_levels)

        if verbose:
            bayes_est[idx] = utils.get_bayes_est(samples[-1], filtered[1])
            max_pos[idx] = utils.get_max_pos(
                filtered[1], bayes_est[idx], samples[-1], filtered_priors)

        par_union += [par for par in filtered[1] if par not in par_union]

    if len(par_union) > 1:
        if not levels:
            levels = [0.68, 0.95]

        sets = corner_plot_settings(levels, samples, one_column, legend_size, fig_width_pt)

        g = plots.get_subplot_plotter(settings=sets)
        g.triangle_plot(
            samples,
            filled=True,
            params=par_union,
            legend_labels=samples_name,
            sharey=True,
            diag1d_kwargs={"normalized": True},
            param_limits=ranges)

        if bhb_prior:
            plot_bhb_prior(g, bhb_prior, levels)

    elif len(par_union) == 1:
        sets = oned_plot_settings()

        if fig_width_pt:
            size = set_size(fig_width_pt, ratio=1)[0]
        elif one_column:
            size = set_size(246, ratio=1)[0]
        else:
            size = set_size(510, ratio=1)[0]

        g = plots.get_single_plotter(settings=sets, ratio=1, width_inch=size)
        g.plot_1d(samples, par_union[0], normalized=True)

        g.add_legend(samples_name, legend_loc="best")

    if any(k_ratio):
        plot_k_bounds(g, samples, k_levels)
    if hpi_levels:
        plot_hpi(g, samples, hpi_points)

    if par_names is None:
        par_names_union = par_union
    else:
        par_names_union = []
        for names in par_names:
            par_names_union += [name for name in names if name not in par_names_union]

    create_ax_labels(par_names_union, labelsize=labelsize)
    g.fig.align_labels()

    # Check if we got the right type, convert to Path if possible
    if not isinstance(plots_dir, Path):
        if isinstance(plots_dir, str):
            plots_dir = Path(plots_dir)
        else:
            err = f"{plots_dir} is not a string or Path object. Falling back to saving in ./plots."
            log.error(err)

    # Make the directory if it doesn't exist
    plots_dir.mkdir(parents=True, exist_ok=True)

    if save and model_name:
        plt.savefig(plots_dir / f"{model_name}_posteriors.pdf", bbox_inches="tight")
    elif save:
        plt.savefig(plots_dir / "posteriors.pdf", bbox_inches="tight")

    if verbose:
        print_stats(k_levels, hpi_points, bayes_est, max_pos, hpi_levels)

    f = plt.gcf()
    axs = f.get_axes()

    return f, axs
