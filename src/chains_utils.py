import os
import re
import sys
import math
import numpy as np
import pandas as pd
from scipy.stats import norm
import importlib.util

from enterprise_extensions import model_utils

import matplotlib.pyplot as plt
from matplotlib import ticker
from matplotlib.ticker import ScalarFormatter
from matplotlib.ticker import FormatStrFormatter
from getdist import plots, MCSamples
import getdist


class ScalarFormatterClass(ScalarFormatter):
   def _set_format(self):
      self.format = "%1.2f"


def optimal_bin_n(data):
    """
    Computes the optimal number of bins following the Freedman-Diaconis rule 
    """
    q1 = np.quantile(data, 0.25)
    q3 = np.quantile(data, 0.75)
    iqr = q3 - q1
    bin_width = (2 * iqr) / (len(data) ** (1 / 3))
    bin_count = int(np.ceil((data.max() - data.min()) / bin_width))

    return bin_count


def set_custom_tick_options(
    ax,
    left=True,
    right=True,
    bottom=True,
    top=True,
    ):
    
    ax.minorticks_on()
    
    ax.tick_params(which='major', direction='in', 
                   length=6, width = 1, 
                   bottom = bottom, 
                   top = top,
                   left = left,
                   right = right,
                   pad = 5)
    ax.tick_params(which='minor',direction='in',
                   length = 3, width = 1, 
                   bottom = bottom, 
                   top = top,
                   left = left,
                   right = right)


def params_loader(file):
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


def vol_2d_contour(sigma):
    return 1 - np.exp(-(sigma**2)/2)


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

    return chain[:, filter_par], params[filter_par]


def create_ax_labels(par_names):

    plt_params = {
        'font.cursive'  : ['Apple Chancery','Textile,Zapf Chancery', 'Sand', 'Script MT','Felipa','cursive'],
        'font.family'  : 'serif',
        'font.serif'  : 'Palatino',
        'font.size'  : 16.0,
        'font.stretch'  : 'normal',
        'font.style'  : 'normal',
        'font.variant'  : 'normal',
        'font.weight'  : 'normal',
        'text.usetex'  : False,
        'mathtext.fontset'  : 'custom',
        'mathtext.rm'  : 'serif',
        'mathtext.it'  : 'serif:italic',
        'mathtext.bf'  : 'serif:bold'}

    plt.rcParams.update(plt_params) 

    labelsize = 25
    ticksize = 25

    f = plt.gcf()
    axarr = f.get_axes()

    N_params = len(par_names)

    if N_params == 1:
        set_custom_tick_options(axarr[0])
        set_custom_tick_options(axarr[0])
        axarr[0].tick_params(axis='x', which='both', labelsize=ticksize - 8)
        axarr[0].set_xlabel(par_names[0],
                        fontsize=labelsize - 9)
        axarr[0].set_ylabel('A', alpha=0) # here just ot prevent padding issues
        fmtr = ticker.ScalarFormatter(useMathText=True)
        axarr[0].xaxis.set_major_formatter(fmtr)

        return

    for idx in range(N_params):
        for idy in range(N_params - idx):
            id = int(N_params * idx + idy - max(idx * (idx - 1) /2, 0))

            axarr[id].tick_params(axis='x', which='both', labelsize=ticksize - 8)

            if idx == 0 and idy == 0:
                set_custom_tick_options(axarr[id])
                axarr[id].tick_params(axis='y', which='both', labelsize=ticksize - 8)
                axarr[id].set_xlabel(par_names[idx], fontsize=labelsize - 8)
                axarr[id].set_ylabel(par_names[N_params - idy - 1],
                              fontsize=labelsize - 8)
            elif idx == N_params - 1:
                set_custom_tick_options(axarr[id], left=False, right=False)
                axarr[id].tick_params(axis='y', which='both', labelsize=ticksize - 8)
                axarr[id].set_xlabel(par_names[idx], fontsize=labelsize - 8)
            elif idy == N_params - idx - 1:
                set_custom_tick_options(axarr[id], left=False, right=False)
            elif idx == 0:
                set_custom_tick_options(axarr[id])
                axarr[id].tick_params(axis='y', which='both', labelsize=ticksize - 8)
                axarr[id].set_ylabel(par_names[N_params - idy - 1],
                              fontsize=labelsize - 8)
            elif idy == 0:
                set_custom_tick_options(axarr[id])
                axarr[id].tick_params(axis='y', which='both', labelsize=ticksize - 8)
                axarr[id].set_xlabel(par_names[idx], fontsize=labelsize - 8)
            else:
                set_custom_tick_options(axarr[id])
                axarr[id].tick_params(axis='y', which='both', labelsize=ticksize - 8)

        fmtr = ticker.ScalarFormatter(useMathText=True)
        axarr[id].xaxis.set_major_formatter(fmtr)
    
    return


def plot_bhb_prior(params, bhb_prior, sigmas):

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
    axarr = f.get_axes()

    N_params = len(params)
    for idx in range(N_params):
        for idy in range(N_params - idx):
            id = int(N_params * idx + idy - max(idx * (idx - 1) /2, 0))
            c2t = N_params - idx - idy - 1

            if str(params[idx]) == 'gw_bhb_0' and str(params[N_params - idy - 1]) == 'gw_bhb_1':
                id_marg_A = int(id + N_params - idx - idy - 1)
                id_marg_g = int(id + c2t*(N_params - idx) - c2t * (c2t + 1) / 2 + c2t)
                for idx in range(len(sigmas)):
                    axarr[id].plot(A_0 + Ell_rot[idx, 0, :] , gamma_0 + Ell_rot[idx, 1, :],
                    color='black', 
                    linestyle='dashed',
                    linewidth=0.7,
                    alpha=0.9)
            elif params[idx] == 'gw_bhb_1' and params[N_params - idy - 1] == 'gw_bhb_0':
                id_marg_g = int(id + N_params - idx - idy - 1)
                id_marg_A = int(id + c2t*(N_params - idx) - c2t * (c2t + 1) / 2 + c2t)
                for idx in range(len(sigmas)):
                    axarr[id].plot(gamma_0 + Ell_rot[idx, 0, :], A_0 + Ell_rot[idx, 1, :],
                    'black',
                    linestyle='dashed',
                    linewidth=0.7,
                    alpha=0.9)
        
    A_pts = np.arange(-17, 15, 0.001)
    g_pts = np.arange(2, 6, 0.001)
    axarr[id_marg_A].plot(A_pts, norm.pdf(A_pts, mu[0], cov[0,0]**(1/2)),
        color='black',
        linestyle='dashed',
        linewidth=1)
    axarr[id_marg_g].plot(g_pts, norm.pdf(g_pts, mu[1], cov[1,1]**(1/2)),
        color='black',
        linestyle='dashed',
        linewidth=1)


def corner_plot_settings(sigmas, samples, one_column):

    sets = plots.GetDistPlotSettings()
    sets.legend_fontsize = 18
    sets.prob_y_ticks = False
    sets.figure_legend_loc = 'upper right'
    sets.linewidth = 2
    sets.norm_1d_density = 'integral'
    sets.alpha_filled_add = 0.8
    if one_column:
        sets.subplot_size_inch = 2
    else:
        sets.subplot_size_inch = 3

    if sigmas is not None:
        sets.num_plot_contours = len(sigmas)
        for sample in samples:
            sample.updateSettings({'contours': [vol_2d_contour(x) for x in sigmas]})

    return sets

def oned_plot_settings():
    sets = plots.GetDistPlotSettings()
    sets.subplot_size_inch = 2
    sets.legend_fontsize = 10
    sets.prob_y_ticks = False
    sets.figure_legend_loc = 'upper right'
    sets.linewidth = 1
    sets.norm_1d_density = 'integral'
    sets.line_styles = ['#006FED',
        '#E03424',
        'gray',
        '#009966',
        '#000866',
        '#336600',
        '#006633',
        'm',
        'r']

    return sets


def plot_posteriors(
    chains,
    params,
    par_to_plot=None,
    par_names=None,
    model_id=None,
    samples_name=None,
    ranges={},
    sigmas=None,
    bhb_prior=False,
    one_column = False,
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

    N_chains = len(chains)

    ranges = params
    params = [np.array(list(par.keys())) for par in params]

    if par_to_plot is None:
        par_to_plot = [None] * N_chains
    if model_id is None:
        model_id = [None] * N_chains
    if not samples_name:
        samples_name = [f'Sample {idx+1}' for idx in range(N_chains)]

    samples = []
    par_union = []
    for idx, chain in enumerate(chains):
        filtered = chain_filter(chain, params[idx], model_id[idx], par_to_plot[idx])
        filtered_ranges = {
            k: v for k, v in ranges[idx].items() if k in filtered[1] and v is not None}

        samples.append(
            MCSamples(
                samples=filtered[0],
                names=filtered[1],
                ranges=filtered_ranges,
                ignore_rows=1))

        par_union += [par for par in filtered[1] if par not in par_union]    
     
    if len(par_union) > 1:
        sets = corner_plot_settings(sigmas, samples, one_column)

        g = plots.get_subplot_plotter(settings=sets)
        g.triangle_plot(samples,
            filled=True, 
            params=par_union, 
            legend_labels=samples_name, 
            diag1d_kwargs={'normalized':True})
        if bhb_prior:
            plot_bhb_prior(par_union, bhb_prior, sigmas)

    elif len(par_union) == 1:
        sets = oned_plot_settings()

        g = plots.get_single_plotter(settings=sets)
        g.plot_1d(samples, par_union[0], normalized=True)

        g.add_legend(samples_name, legend_loc='best')

        for idx, sample in enumerate(samples):
            plt.hist(sample.samples,
                density=True,
                bins=optimal_bin_n(sample.samples),
                alpha=0.1,
                color=sets.line_styles[idx])

    if par_names is None:
        par_names_union = par_union
    else:
        par_names_union = []
        for names in par_names:
            par_names_union += [name for name in names if name not in par_names_union] 
    
    create_ax_labels(par_names_union)

    if save and model_name:
        plt.savefig(f'./plots/{model_name}_posteriors.pdf', bbox_inches="tight")
    elif save:
        plt.savefig('./plots/posteriors.pdf', bbox_inches="tight")

    return

