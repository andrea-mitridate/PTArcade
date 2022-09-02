import os
import sys
import math
import numpy as np
import pandas as pd
from cycler import cycler

import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
from matplotlib.ticker import FormatStrFormatter
import matplotlib as mpl
import seaborn as sns
import corner

class ScalarFormatterClass(ScalarFormatter):
   def _set_format(self):
      self.format = "%1.2f"

colors = sns.color_palette("tab10")


def import_chains(chains_dir, burn_frac=1/4):
    """
    Import the chains and ther parameter file removing the burn-in region.
    :param chains_dir: name of the directory containing the chains.
    :param burn_frac: fraction of the chain that is removed from his head.
    """

    # get all the chain directories 
    directories = [x for x in os.listdir(chains_dir) if not x.startswith(".")]

    # get the parameter lits and check that all the chains have the same parameters
    params = np.loadtxt(os.path.join(chains_dir, directories[0], 'pars.txt'), dtype=str)

    for chain in directories:
        temp_par = np.loadtxt(os.path.join(chains_dir, chain, 'pars.txt'), dtype=str)

        if (temp_par != params).all():
            sys.exit(" ERROR: chains have differnt parameters.")

    # add name for sampler parameters 
    params = np.concatenate(
        (params,
        ['log_posterior', 'log_likelihood', 'acceptance_rate', 'n_parall']))

    # import and merge all the chains removing the burn-in
    for idx, dir in enumerate(directories):
        if idx == 0:
            mrgd_chain = pd.read_csv(
                os.path.join(chains_dir, dir, 'chain_1.txt'),
                sep='\t',
                names=params)
            mrgd_chain = mrgd_chain.iloc[int(len(mrgd_chain) * burn_frac):]

        temp_chain = pd.read_csv(
            os.path.join(chains_dir, dir, 'chain_1.txt'),
            sep='\t',
            names=params)
        mrgd_chain = pd.concat(
            [mrgd_chain, temp_chain.iloc[int(len(temp_chain) * burn_frac):]],
            ignore_index=True,
            sort=False)

    mrgd_chain = mrgd_chain.to_numpy()

    return params, mrgd_chain


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


def plot_posteriors(
    chain,
    params,
    params_name=None,
    model_id = None,
    ranges=None,
    smooth=1.5,
    sigmas=None,
    save=False,
    model_name=None):
    """
    Plot a corner plot with the posterior distributions.

    chain: numpy array
        chain to be plotted
    params: list
        list of parameters names 
    params_names: dict (optional)
        dictionary with as keys the names of the parameters in the params list
        to be plotted, and with values the formatted parameters name to be shown
        in the plots. (default is None, and in this case  plots all
        common parameters + the mcmc parameters are shown without any formatting.
    model_id: 0 or 1 (optional)
        assuming that the data are generated using hypermodels, specifies the model
        for which to plot the posteriors (default is None and in this case nmodel
        is taken to be 1)
    ranges: list of lists (optional)
        containing the ranges for the parameters to plot (default is None, in this 
        case the whole range of the data is shown)
    smooth: int (optional)
        The standard deviation for Gaussian kernel passed to 
        scipy.ndimage.gaussian_filter to smooth the 2-D and 1-D histograms
        respectively. If None (default), no smoothing is applied.
    save: boolean (optional) 
        if set to True the plot is going to be saved in the folder
        "./plots/" (default is False, and in this case not plot is saved)
    model_name: str (required if save is set to True)
        model name used to name the output files
    """

    # define the filter for the model to plot 
    nmodel_idx = list(params).index('nmodel')

    if model_id is None:
        print("No model ID specified, posteriors are plotted for model 1")
        filter = chain[:, nmodel_idx] > 0.5
    elif model_id == 0:
        filter = chain[:, nmodel_idx] < 0.5
    elif model_id == 1:
        filter = chain[:, nmodel_idx] > 0.5
    else:
        sys.exit(" ERROR: model_idx can only be an integer equal to 0 or 1")

    # define the list of parameters to plot and find therir position in the chains
    params_dic = {}

    if params_name:
        for par, name in params_name.items():
            try:
                params_dic[name] = list(params).index(par)
            except:
                print(f"WARNING: {par} does not appear in the parameter list")

    else:
        params = [
            x
            for x in params
            if x
            not in [
                "nmodel",
                "log_posterior",
                "log_likelihood",
                "acceptance_rate",
                "n_parall"]]

        for idx, par in enumerate(params):
            if '+' not in par and '-' not in par:
                params_dic[par] =  idx

    # gets the data for the plot
    corner_data = []
    labels = []
    for par, pos in params_dic.items():
        corner_data.append(chain[filter, pos])
        labels.append(par)
    
    corner_data = np.stack(list(corner_data), axis=-1)

    # if only one parameter needs to be plotted create the histogram
    if len(corner_data[0]) == 1:
        plt.hist(
            corner_data, 
            bins=50, 
            density=True, 
            histtype='stepfilled', 
            lw=2, 
            color='C0', 
            alpha=0.5)

        plt.xlabel(labels[0])
        plt.ylabel('PDF')
        if ranges:
            plt.xlim(ranges[0])
        plt.tight_layout()

    # if multiple parameters need to be plotted creates a corner plot 
    else:
        if sigmas is None:
            levels = (vol_2d_contour(2),vol_2d_contour(1))
        else:
            levels = list([vol_2d_contour[x] for x in sigmas])

        corner.corner(
            corner_data,
            labels=labels,
            bins=50,
            color='steelblue',
            smooth=smooth,
            alpha=1,
            plot_datapoints=False,
            plot_contours=True,
            levels=levels,
            range=ranges,
            plot_density=True
            )

    if save:
        if model_name:
            plt.savefig(f"./plots/{model_name}_corner.pdf")
        else:
            print('Please specify a model name to save the corner plot.')

    return 
