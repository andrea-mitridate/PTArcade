import os
import re
import sys
import math
import numpy as np
import pandas as pd
import warnings
from scipy.stats import norm
from scipy import integrate
from itertools import combinations
from scipy.optimize import minimize, Bounds

from enterprise_extensions import model_utils

import matplotlib.pyplot as plt
from matplotlib import ticker
from matplotlib.ticker import ScalarFormatter
from matplotlib.ticker import FormatStrFormatter
from getdist import plots, MCSamples



plt_params = {
        'axes.linewidth' : 0.5,
        'text.usetex'  : True,
        }

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


class ScalarFormatterClass(ScalarFormatter):
   def _set_format(self):
      self.format = "%.1f"


def set_size(width, fraction=1, ratio=None, subplots=(1, 1)):
    """Set figure dimensions to avoid scaling in LaTeX.

    Parameters
    ----------
    width: float
            Document textwidth or columnwidth in pts
    fraction: float, optional
            Fraction of the width which you wish the figure to occupy

    Returns
    -------
    fig_dim: tuple
            Dimensions of figure in inches
    """
    # Width of figure (in pts)
    fig_width_pt = width * fraction

    # Convert from pt to inches
    inches_per_pt = 1 / 72.27

    # Figure width in inches
    fig_width_in = fig_width_pt * inches_per_pt
    # Figure height in inches
    if not ratio :
        # Golden ratio to set aesthetic figure height
        # https://disq.us/p/2940ij3
        ratio = (5**.5 - 1) / 2

    fig_height_in = fig_width_in * ratio * (subplots[0] / subplots[1])

    fig_dim = (fig_width_in, fig_height_in)

    return fig_dim


def set_custom_tick_options(
    ax,
    left=True,
    right=True,
    bottom=True,
    top=True,
    width=0.5,
    length=2.5
    ):
    '''
    Sets custom tick options for a matplotlib axis.

    Parameters
    ----------
    ax (matplotlib axis): 
        The axis to which the tick options will be applied.
    left (bool): 
        If True, left ticks will be visible.
    right (bool):
        If True, right ticks will be visible.
    bottom (bool):
        If True, bottom ticks will be visible.
    top (bool):
        If True, top ticks will be visible.

    Returns
    -------
        None
    '''
    
    ax.minorticks_on()
    
    ax.tick_params(which='major', direction='in', 
                   length=length, width = width, 
                   bottom = bottom, 
                   top = top,
                   left = left,
                   right = right,
                   pad = 2)
    ax.tick_params(which='minor',direction='in',
                   length = length/2, width = width, 
                   bottom = bottom, 
                   top = top,
                   left = left,
                   right = right)


def params_loader(file):
    '''
    Loads a parameter file and returns a dictionary of the parameters with their respective values or None.

    Parameters
    ----------
    file (str): 
        The path to the prior.txt file.

    Returns
    -------
    dic
        A dictionary with the parameters prior ranges.
    '''

    params = {}

    with open(file) as f:
        for line in f:
            if ':' not in line:
                continue

            key = line.split(":")[0]

            if "Uniform" in line:
                min = float(re.search('pmin=(.*?),', line).group(1))
                max = float(re.search('pmax=(.*?)\\)', line).group(1))

                params[key] = (min, max)
            
            elif "Normal" in line:
                dim = len(re.search('\\[(.*?)\\]', line).group(1).split('    '))
                for i in range(dim):
                    params[f"{key}_{i}"] = None

            else:
                params[key] = None

    return params


def import_chains(chains_dir, burn_frac=1/4):
    """
    Import the chains and ther parameter file removing the burn-in region.
    :param chains_dir: name of the directory containing the chains.
    :param burn_frac: fraction of the chain that is removed from his head.
    """

    # get all the chain directories 
    directories = [x for x in os.listdir(chains_dir) if not x.startswith(".")]

    # get the parameter lits and check that all the chains have the same parameters
    params = params_loader(os.path.join(chains_dir, directories[0], 'priors.txt'))

    for chain in directories:
        temp_par = params_loader(os.path.join(chains_dir, chain, 'priors.txt'))

        if temp_par != params:
            sys.exit(" ERROR: chains have differnt parameters.")

    # add name for sampler parameters 

    params.update({
        'nmodel' : None,
        'log_posterior' : None,
        'log_likelihood' : None,
        'acceptance_rate' : None,
        'n_parall' : None
        })

    # import and merge all the chains removing the burn-in
    for idx, dir in enumerate(directories):
        if idx == 0:
            mrgd_chain = pd.read_csv(
                os.path.join(chains_dir, dir, 'chain_1.txt'),
                sep='\t',
                names=list(params.keys()))
            mrgd_chain = mrgd_chain.iloc[int(len(mrgd_chain) * burn_frac):]

        temp_chain = pd.read_csv(
            os.path.join(chains_dir, dir, 'chain_1.txt'),
            sep='\t',
            names=list(params.keys()))
        mrgd_chain = pd.concat(
            [mrgd_chain, temp_chain.iloc[int(len(temp_chain) * burn_frac):]],
            ignore_index=True,
            sort=False)

    mrgd_chain = mrgd_chain.dropna()

    mrgd_chain = mrgd_chain.to_numpy(dtype=float)

    return params, mrgd_chain


def compute_bf(chain, params):
    """
    Computes the Bayes factor for model 0 vs model 1
    :param chain: numpy array containing the chain to be plotted
    :param params: list with the names of the parameters appearing in the chain
    """

    # define the filter for the model to plot 
    nmodel_idx = list(params).index('nmodel')

    bf, unc = model_utils.odds_ratio(chain[:, nmodel_idx], models=[0,1])

    return bf, unc


def plot_chains(chain, params, params_name=None, save=False, model_name=None):
    """
    Plot the chain for the parameters specified by params.
    :param chain: numpy array containing the chain to be plotted
    :param params: list with the names of the parameters appearing in the chain
    :param params_names: dictionary with keys the names of the parameters in the
        params list to be plotted, and with values the formatted parameters name 
        to be shown in the plots.
        [default: None] plots all the commong parameters + the mcmc parameters 
        without any formatting
    :params save: boolean, if set to True the plot is going to be saved in the folder 
        "./plots/"
        [default:False]
    :params model_name: string with the model name. Used to name the output files
        [default:None]
    """

    params_dic = {}
    
    if params_name:
        for idx, par in enumerate(params):
            if par in params_name.keys():
                formatted_name = params_name[par]
                params_dic[formatted_name] = idx
        
        if len(params_dic) != len(params_name):
            print("WARNING: some of the requested parameters does not appear in the parameter list")

    else:
        for idx, par in enumerate(params):
            if '+' not in par and '-' not in par:
                params_dic[par] =  idx
    
    n_par = len(params_dic)
    n_row = int(n_par**0.5)
    n_col=int(math.ceil(n_par / float(n_row)))

    fig, ax = plt.subplots(n_row, n_col, figsize=(5 * n_col, 2.5 * n_row))

    for idx, par in enumerate(params_dic.keys()):
        x = math.floor(idx / n_row)
        y = idx % n_row
        
        ax[y,x].plot(chain[:, params_dic[par]])
        ax[y,x].set_ylabel(par, fontsize=15)
        yScalarFormatter = ScalarFormatterClass(useMathText=True)
        yScalarFormatter.set_powerlimits((-1,2))
        ax[y,x].yaxis.set_major_formatter(yScalarFormatter)

    plt.gca().xaxis.set_major_formatter(FormatStrFormatter('%g'))

    plt.tight_layout()
    plt.show()

    if save:
        if model_name:
            plt.savefig(f"./plots/{model_name}_chains.pdf")
        else:
            print('Please specify a model name to save the chain plot.')

    return


def level_to_sigma(level):
    return np.sqrt(-2 * np.log(1-level))


def chain_filter(chain, params, model_id, par_to_plot):
    """
    Select the rows in the chain that correspond the model and paramerers 
    we want to plot the posteriors for.
    """
    nmodel_idx = list(params).index('nmodel')

    if model_id is None:
        print("No model ID specified, posteriors are plotted for model 1")
        filter_model = chain[:, nmodel_idx] > 0.5
    elif model_id == 0:
        filter_model = chain[:, nmodel_idx] < 0.5
    elif model_id == 1:
        filter_model = chain[:, nmodel_idx] > 0.5
    else:
        sys.exit(" ERROR: model_idx can only be an integer equal to 0 or 1")

    chain = chain[filter_model]

    if par_to_plot:
        filter_par = [list(params).index(x) for x in par_to_plot if x in params]
    else:
        filter_par = [list(params).index(x)
                for x in params 
                if x
                not in [
                    "nmodel",
                    "log_posterior",
                    "log_likelihood",
                    "acceptance_rate",
                    "n_parall"]
                and '+' not in x
                and '-' not in x]

    filtered_par = [par.replace('_','-') for par in params[filter_par]]

    return chain[:, filter_par], filtered_par


def create_ax_labels(par_names):

    N_params = len(par_names)

    f = plt.gcf()
    axs = f.get_axes()

    if N_params == 1:
        labelsize = 10
        ticksize = 10

        set_custom_tick_options(axs[0], width=0.5, length=5)
        axs[0].tick_params(axis='x', which='both', labelsize=ticksize)
        axs[0].tick_params(axis='y', which='both', labelsize=ticksize)
        axs[0].set_xlabel(par_names[0], fontsize=labelsize, labelpad=10)
        axs[0].set_ylabel('A', alpha=0) # here just ot prevent padding issues

        return

    labelsize = 6
    ticksize = 6
    
    # do this loop using combinations_with_replacement from itertools 
    for idx in range(N_params):
            for idy in range(N_params - idx):
                id = int(N_params * idx + idy - max(idx * (idx - 1) /2, 0))

                set_custom_tick_options(axs[id], width=0.5)
                axs[id].tick_params(axis='x', which='both', labelsize=ticksize)

                if idx == 0 and idy == 0:
                    set_custom_tick_options(axs[id], width=0.5)
                    axs[id].tick_params(axis='y', which='both', labelsize=ticksize)
                    axs[id].set_xlabel(par_names[idx], fontsize=labelsize, labelpad=7)
                    axs[id].set_ylabel(par_names[N_params - idy - 1],
                                fontsize=labelsize)
                elif idx == N_params - 1:
                    set_custom_tick_options(axs[id], left=False, right=False, width=0.5)
                    axs[id].tick_params(axis='y', which='both', labelsize=ticksize)
                    axs[id].set_xlabel(par_names[idx], fontsize=labelsize, labelpad=7)
                elif idy == N_params - idx - 1:
                    set_custom_tick_options(axs[id], left=False, right=False, width=0.5)
                elif idx == 0:
                    set_custom_tick_options(axs[id], width=0.5)
                    axs[id].tick_params(axis='y', which='both', labelsize=ticksize)
                    axs[id].set_ylabel(par_names[N_params - idy - 1],
                                fontsize=labelsize)
                elif idy == 0:
                    set_custom_tick_options(axs[id], width=0.5)
                    axs[id].tick_params(axis='y', which='both', labelsize=ticksize)
                    axs[id].set_xlabel(par_names[idx], fontsize=labelsize, labelpad=7)
                else:
                    set_custom_tick_options(axs[id], width=0.5)
                    axs[id].tick_params(axis='y', which='both', labelsize=ticksize)
    
    return


def plot_bhb_prior(params, bhb_prior, levels):

    sigmas = [level_to_sigma(level) for level in levels]

    if bhb_prior == 'NG15':
        mu = np.array([-15.1151815, 4.34183987])
        cov = np.array([[0.0647048, 0.00038692], [0.00038692, 0.07741015]])
    elif bhb_prior == 'IPTA2':
        mu = np.array([-15.02928454, 4.14290127])
        cov = np.array([[0.06869369, 0.00017051], [0.00017051, 0.04681747]])
        
    A_0, gamma_0 = mu

    eig = np.linalg.eig(np.linalg.inv(cov))
    a, b = eig[0]
    R_rot =  eig[1]

    t = np.linspace(0, 2*np.pi, 100)
    Ell = - np.array([(a)**(-1/2) * np.cos(t) , (b)**(-1/2) * np.sin(t)])

    Ell_rot = np.zeros((len(sigmas), 2, Ell.shape[1]))
    for idx, sigma in enumerate(sigmas):
        for i in range(Ell.shape[1]):
            Ell_rot[idx, :, i] = np.dot(R_rot, sigma * Ell[:,i])

    f = plt.gcf()
    axs = f.get_axes()

    lw = 0.45

    N_params = len(params)
    #just get the axes for the bhb parameters using plot.get_param_array(sample, [par_1, par_2])
    for idx in range(N_params):
        for idy in range(N_params - idx):
            id = int(N_params * idx + idy - max(idx * (idx - 1) /2, 0))
            c2t = N_params - idx - idy - 1

            if str(params[idx]) == 'gw-bhb-0' and str(params[N_params - idy - 1]) == 'gw-bhb-1':
                id_marg_A = int(id + N_params - idx - idy - 1)
                id_marg_g = int(id + c2t*(N_params - idx) - c2t * (c2t + 1) / 2 + c2t)
                for idx in range(len(sigmas)):
                    axs[id].plot(A_0 + Ell_rot[idx, 0, :] , gamma_0 + Ell_rot[idx, 1, :],
                    color='black', 
                    linestyle='dashed',
                    linewidth=lw,
                    alpha=0.9)
            elif params[idx] == 'gw-bhb-1' and params[N_params - idy - 1] == 'gw-bhb-0':
                id_marg_g = int(id + N_params - idx - idy - 1)
                id_marg_A = int(id + c2t*(N_params - idx) - c2t * (c2t + 1) / 2 + c2t)
                for idx in range(len(sigmas)):
                    axs[id].plot(gamma_0 + Ell_rot[idx, 1, :], A_0 + Ell_rot[idx, 0, :],
                    'black',
                    linestyle='dashed',
                    linewidth=lw,
                    alpha=0.9)
        
    A_pts = np.arange(-17, -13, 0.001) 
    g_pts = np.arange(2, 6, 0.001)
    axs[id_marg_A].plot(A_pts, norm.pdf(A_pts, mu[0], cov[0,0]**(1/2)),
        color='black',
        linestyle='dashed',
        linewidth=lw)
    axs[id_marg_g].plot(g_pts, norm.pdf(g_pts, mu[1], cov[1,1]**(1/2)),
        color='black',
        linestyle='dashed',
        linewidth=lw)


def corner_plot_settings(levels, samples, one_column):

    sets = plots.GetDistPlotSettings()
    sets.prob_y_ticks = False
    sets.figure_legend_loc = 'upper right'
    sets.norm_1d_density = 'integral'
    sets.alpha_filled_add = 0.8
    sets.legend_fontsize = 12
    sets.linewidth = 2
    if one_column:
        sets.fig_width_inch = set_size(246, ratio=1)[0]
    else:
        sets.fig_width_inch = set_size(510, ratio=1, fraction=0.65)[0]

    if levels is not None:
        sets.num_plot_contours = len(levels)
        for sample in samples:
            sample.updateSettings({'contours': levels})

    return sets


def oned_plot_settings():
    sets = plots.GetDistPlotSettings()

    sets.legend_fontsize = 9
    sets.prob_y_ticks = False
    sets.linewidth = 1.1
    sets.figure_legend_loc = 'upper right'
    sets.norm_1d_density = 'integral'
    sets.line_styles = colors

    return sets


def bisection(f, a, b, tol): 
    """"
    approximates a root of f bounded 
    by a and b to within tolerance 
    |f(m)| < tol with m the midpoint 
    between a and b; Recursive implementation
    """
    
    # check if a and b bound a root
    if np.sign(f(a)) == np.sign(f(b)):
        return None
        
    # get midpoint
    m = (a + b)/2
    
    if np.abs(f(m)) < tol:
        # stopping condition, report m as root
        return m
    elif np.sign(f(a)) == np.sign(f(m)):
        # case where m is an improvement on a. 
        # make recursive call with a = m
        return bisection(f, m, b, tol)
    elif np.sign(f(b)) == np.sign(f(m)):
        # case where m is an improvement on b. 
        # make recursive call with b = m
        return bisection(f, a, m, tol)


def k_ratio_aux_1D(
    sample, 
    BF, 
    par,
    par_range,
    k_ratio):
    """"
    Return unnormalized 1D posterior density, corresponding normalization and k_ratio bound
    sample: MCSamples instance
    BF: float
        Bayes factor for exotic + SMBHB vs. SMBHB model
    par: string
        name of the two parameters for which the K-ratio bound should be plotted
    par_range: list
        lower and upper prior limits
    k_ratio: float
        Fraction of plateau height at which height level is determined
    """
    
    density1D = MCSamples.get1DDensity(sample, par)
    
    norm = integrate.quad(density1D, par_range[0], par_range[1], full_output=1)[0]
    
    prior = 1/(par_range[1]-par_range[0])
    
    height_KB = k_ratio*prior/BF*norm
    
    k_val = bisection(f = (lambda x: density1D(x)-height_KB), a = par_range[0], b = par_range[1], tol = 10**(-8))

    return k_val

def k_ratio_aux_2D(
    sample, 
    BF, 
    par_1, 
    par_2, 
    par_range_1, 
    par_range_2, 
    k_ratio):
    """"
    Return unnormalized 2D posterior density, corresponding normalization and height level corresponding 
    to given k_ratio
    sample: MCSamples instance
    BF: float
        Bayes factor for exotic + SMBHB vs. SMBHB model
    par_1/2: string
        name of the two parameters for which the K-ratio bound should be plotted
    par_range_1/2: list
        lower and upper prior limits
    k_ratio: float (optional)
        Fraction of plateau height at which height level is determined
    """
    density2D = MCSamples.get2DDensity(sample, par_1, par_2)
    
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        norm = integrate.dblquad(density2D, par_range_2[0], par_range_2[1], par_range_1[0], par_range_1[1])[0]#Calculate prior value at each point
    
    prior = 1/(par_range_2[1]-par_range_2[0]) * 1/(par_range_1[1]-par_range_1[0])
    
    height_KB = k_ratio*prior/BF*norm
    return height_KB


def get_bayes_est(samples, params):
    #samps is MCSamples instance
    x = list(samples.setMeans())
    xerr = list(np.sqrt(samples.getVars()))
    x = zip(x, xerr)
    x = dict(zip(params, x))
    return x


def get_max_pos(params, bayes_est, sample, priors, spc=10):

    out = {}
    
    for par in params:
        bounds = priors.get(par)

        density =  MCSamples.get1DDensity(sample, par, normalized=False)

        mind = lambda x: -density(x)[0]
        if not bounds:
            x = minimize(mind, np.array(bayes_est.get(par)[0]))
            x0 = x.x
            d0 = mind(x0)
        else:
            ini = np.linspace(bounds[0], bounds[1], spc)
            ini = [val for val in ini]
            ini.append(bayes_est.get(par)[0])
            x = []
            for i in range(len(ini)):
                x1 = minimize(mind, ini[i])
                x.append(x1.x)
                bnds = Bounds(bounds[0], bounds[1])
                x2 = minimize(mind, ini[i], bounds=bnds)
                x.append(x2.x)
            d0 = 0
            x0 = 0
            for val in x:
                if(mind(val)<d0):
                    x0 = val
                    d0 = mind(val)

        out[par] = x0[0]

    return out


def get_k_levels(sample, pars, priors, bf, k_ratio):
    np_pars = [p for p in pars if p not in ["gw-bhb-0", "gw-bhb-1"]]

    levels_2d = []
    for par_1, par_2 in combinations(np_pars, r=2):
        levels_2d.append(
            [par_1,
             par_2,
             k_ratio_aux_2D(
                sample=sample,
                BF=bf,
                par_1=par_1,
                par_2=par_2,
                par_range_1=priors[par_1],
                par_range_2=priors[par_2],
                k_ratio=k_ratio)])

    levels_1d = []
    for par in np_pars:
        levels_1d.append(
            [par,
            k_ratio_aux_1D(
                sample = sample,
                BF = bf,
                par = par,
                par_range = priors[par],
                k_ratio = k_ratio)])

    return np.array(levels_1d), np.array(levels_2d)


def get_c_levels(sample, pars, levels):
    
    hpi = []
    for par in pars: 
        density = MCSamples.get1DDensity(sample, par)

        points = [[]] * len(levels)
        for idx, level in enumerate(levels):
            x = density.getLimits(level)
            if not x[-1] and not x[-2]:
                points[idx] = [x[0], x[1]]
            elif not x[-1]:
                points[idx] = [False, x[1]]
            elif not x[-2]:
                points[idx] = [x[0], False]
            else:
                points[idx] = [False, False]
        
        hpi.append([par, points])

    return np.array(hpi, dtype='object')


def plot_k_bounds(plot, samples, k_levels):

    for idx, sample in enumerate(samples):
        if k_levels[idx]:    
            for x in k_levels[idx][0]:
                if x[-1]:
                    par = x[0]
                    p0 = float(x[-1])
                    if len(k_levels[idx][0]) == 1:
                        plt.plot([p0, p0], [0,10], color=colors[idx], alpha=0.8, lw=0.8)
                    else:
                        ax = plot.get_axes_for_params(par)
                        ax.plot([p0, p0], [0,10], color=colors[idx], alpha=0.8, lw=0.8)
                    
            for x in k_levels[idx][1]:
                par_1 = x[0]
                par_2 = x[1]
                level = float(x[-1])
                par_1, par_2 = plot.get_param_array(sample, [par_1, par_2])
                ax = plot.get_axes_for_params(par_1, par_2)

                density = plot.sample_analyser.get_density_grid(sample, par_1, par_2,
                                                            conts=plot.settings.num_plot_contours,
                                                            likes=plot.settings.shade_meanlikes)
                
                levels = sorted(np.append([density.P.max() + 1], level))
                cs = ax.contourf(density.x, density.y, density.P, levels, colors='#ffffff00', alpha=0.1, extend = 'both')
                cs.cmap.set_over('#ffffff00')
                cs.cmap.set_under('gray')
                cs.changed()

                levels = sorted(np.append([density.P.max() + 1], level))
                ax.contour(density.x, density.y, density.P,
                        levels = [level],
                        colors='grey',
                        linewidths=0.5,
                        alpha=1) 
   
    return


def plot_hpi(plot, samples, hpi_points):

    lw=0.5

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
                    ax.plot([x1, x1], [0, p1], ls = 'dashed', color = colors[idx], lw=lw)
                if x2:
                    p2 = density(x2)[0]
                    ax.plot([x2, x2], [0, p2], ls = 'dashed', color = colors[idx], lw=lw)

    return


def print_stats(k_levels, hpi_points, bayes_est, max_pos, levels):

    for idx in np.arange(len(k_levels)):
        print(f'----- Stats for sample #{idx}-----')
        
        if any(x != [] for x in k_levels[idx]):
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
    chains,
    params,
    par_to_plot=None,
    par_names=None,
    model_id=None,
    samples_name=None,
    hpi_levels = [0.68, 0.95],
    k_ratio = None,
    ranges={},
    levels=None,
    bhb_prior=False,
    one_column = False,
    verbose = False,
    save=False,
    model_name=None):
    """"
    Plot posterior distributions for the chosen parameteres in the chains. 
    chains: list of lists
        contains all the chains to be plotted
    params: list of lists
        contains the name of the parameters appearing in the chains
    par_to_plot: list of lists (optional)
        contains the paramter to plot from each chain
    par_names: lists of list (optional)
        contains the LaTeX formatted names for the plotted parameters 
        of each model. If not specified the name in the par files will be used.
    model_id: 0 or 1 (optional)
        assuming that the data are generated using hypermodels, specifies the model
        for which to plot the posteriors (default is None and in this case nmodel
        is taken to be 1)
    samples_name: list of strings
        contains the name of the models associated to each chains, and it's used to
        create the legend . If not specified, the labels in the legend will be
        Sample 1, Sample 2, ...
    ranges: list of lists (optional)
        containing the ranges for the parameters to plot (default is None, in this 
        case the whole range of the data is shown)
    save: boolean (optional) 
        if set to True the plot is going to be saved in the folder
        "./plots/" (default is False, and in this case not plot is saved)
    model_name: str (optional)
        model name used to name the output files. If not specified the plot will be
        saved as 'corner.pdf'
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
    if not samples_name:
        samples_name = [f'Sample {idx+1}' for idx in range(N_chains)]

    samples = []
    par_union = []
    k_levels = [[]] * N_chains
    hpi_points = [[]] * N_chains
    bayes_est = [[]] * N_chains
    max_pos = [[]] * N_chains
    for idx, chain in enumerate(chains):
        filtered = chain_filter(chain, params[idx], model_id[idx], par_to_plot[idx])
        filtered_priors = {
            k.replace("_", "-"): v
            for k, v in priors[idx].items()
            if k.replace("_", "-") in filtered[1] and v is not None
        }

        samples.append(
            MCSamples(
                samples=filtered[0],
                names=filtered[1],
                ranges=filtered_priors,
                ignore_rows=1))
        
        if k_ratio[idx]:
            bf = compute_bf(chain, params[idx])[0]
            k_levels[idx] = get_k_levels(
                sample=samples[-1],
                pars=filtered[1],
                priors=filtered_priors,
                bf=bf,
                k_ratio=k_ratio[idx])
        else:
            k_levels[idx] = None
            
        if hpi_levels:
            hpi_points[idx] = get_c_levels(
                sample=samples[-1],
                pars=filtered[1],
                levels=hpi_levels)
            
        if verbose:
            bayes_est[idx] = get_bayes_est(samples[-1], filtered[1])
            max_pos[idx] = get_max_pos(filtered[1], bayes_est[idx], samples[-1], filtered_priors)

        par_union += [par for par in filtered[1] if par not in par_union]    
        

    if len(par_union) > 1:
        sets = corner_plot_settings(levels, samples, one_column)        
        
        g = plots.get_subplot_plotter(settings=sets)
        g.triangle_plot(samples,
            filled=True, 
            params=par_union, 
            legend_labels=samples_name, 
            sharey = True,
            diag1d_kwargs={'normalized':True},
            param_limits=ranges)
        if bhb_prior:
            plot_bhb_prior(par_union, bhb_prior, levels)
        

    elif len(par_union) == 1:
        sets = oned_plot_settings()

        if one_column:
            size = set_size(246, ratio=1)[0]
        else:
            size = set_size(510, ratio=1)[0]

        g = plots.get_single_plotter(settings=sets, 
                                    ratio=1, 
                                    width_inch=size)
        g.plot_1d(samples, par_union[0], normalized=True)
        
        g.add_legend(samples_name, legend_loc='best')

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

    create_ax_labels(par_names_union)
    g.fig.align_labels()

    if save and model_name:
        plt.savefig(f'./plots/{model_name}_posteriors.pdf', bbox_inches="tight")
    elif save:
        plt.savefig('./plots/posteriors.pdf', bbox_inches="tight")

    if verbose:
        print_stats(k_levels, hpi_points, bayes_est, max_pos, hpi_levels)

    f = plt.gcf()
    axs = f.get_axes()
        
    return f, axs
