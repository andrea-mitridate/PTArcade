import os
import re
import sys
import time
import acor
import warnings
import numpy as np
import pandas as pd
import pyarrow as pa
from scipy import integrate
import pyarrow.feather as pf
import pyarrow.parquet as pq
from getdist import MCSamples
from itertools import combinations
from astroML.resample import bootstrap
from scipy.optimize import minimize, Bounds
from enterprise_extensions import model_utils


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
                dim = len(re.search('\\[(.*?)\\]', line).group(1).split())
                for i in range(dim):
                    params[f"{key}_{i}"] = None

            else:
                params[key] = None

    return params


def import_chains(chains_dir, burn_frac=1/4, quick_import=True, chain_ext=".txt"):
    """
    Import the chains and their parameter file.

    Parameters:
    ----------
    chains_dir : str
        Name of the directory containing the chains.
    burn_frac : float, optional
        Fraction of the chain that is removed from the head (default is 1/4).
    quick_import : bool, optional
        Flag to skip importing the rednoise portion of chains (default is True).
    chain_ext : str, optional
        The file extension of the chain files. Compressed files can be used (default is ".txt").

    Returns:
    -------
    params : dict
        Dictionary containing the parameter names and their values.
    mrgd_chain : numpy.ndarray
        Numpy array containing the merged chains without the burn-in region.

    Raises:
    ------
    sys.exit : Exception
        Raised when the chains have different parameters.
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
    name_list = list(params.keys())
    if quick_import:
        # Search reversed list for first occurence of "red_noise"
        # Return the index (remember, the list is reversed!)
        # The use of `next` and a generator makes it so that we don't have to
        # search the whole list, we stop when we get the first match
        red_noise_ind = next((i for i in enumerate(name_list[::-1]) if "red_noise" in i[1]))[0]

        # Slice the list so that we begin directly after the index found above
        usecols = name_list[-1*red_noise_ind:]

        params = {name: params[name] for name in usecols}
    else:
        usecols = name_list
    dtypes = {name: float for name in usecols}
    # import and merge all the chains removing the burn-in

    print("Starting import from", chains_dir)
    start_time = time.time()

    if chain_ext==".feather":
        table_list = []
        for dir in directories:
            table = pf.read_table(os.path.join(chains_dir, dir, 'chain_1' + chain_ext), columns=usecols)
            table_list.append(table.slice(offset=int(table.num_rows * burn_frac)).drop_null())


        mrgd_chain = pa.concat_tables(table_list).to_pandas()

    elif chain_ext==".parquet":
        table_list = []
        for dir in directories:
            table = pq.read_table(os.path.join(chains_dir, dir, 'chain_1' + chain_ext), columns=usecols)
            table_list.append(table.slice(offset=int(table.num_rows * burn_frac)).drop_null())

        mrgd_chain = pa.concat_tables(table_list).to_pandas()


    else:
        mrgd_chain = pd.concat((pd.read_csv(
            os.path.join(chains_dir, dir, 'chain_1' + chain_ext),
            sep='\t',
            names=name_list,
            dtype=dtypes,
            usecols=usecols).iloc[lambda x: int(len(x) * burn_frac) <= x.index]
                                for dir in directories),
                               ignore_index=True,
                               sort=False)
        mrgd_chain = mrgd_chain.dropna()

    mrgd_chain = mrgd_chain.to_numpy(dtype=float)

    print(f"Finished importing   {chains_dir} in {time.time() - start_time:.2f}s")
    return params, mrgd_chain


def chain_filter(chain, params, model_id, par_to_plot):
    """
    This function filters the rows in the provided chain according to the specified model and parameters. It selects
    rows that correspond to the specified model ID and parameters to plot their posteriors.

    Parameters
    ----------
    chain : numpy.ndarray
        The Markov Chain Monte Carlo (MCMC) chain to be filtered. This should be a multi-dimensional array where each
        row represents a state in the chain, and each column represents a parameter.

    params : list of str
        The names of the parameters in the chain. This should be a list of strings with the same length as the number
        of columns in the chain.

    model_id : int or None
        The ID of the model to filter the chain for. This should be either 0 or 1. If None, the function will select
        rows for model 0.

    par_to_plot : list of str or None
        The names of the parameters to filter the chain for. If None, the function will select all parameters except
        'nmodel', 'log_posterior', 'log_likelihood', 'acceptance_rate', and 'n_parall', and parameters containing '+' or '-'.

    Returns
    -------
    tuple
        A tuple containing the filtered chain and the list of filtered parameters:
        
        - chain: numpy.ndarray
            The filtered chain, containing only rows corresponding to the specified model ID and parameters.
        
        - filtered_par: list of str
            The list of filtered parameter names.

    Raises
    ------
    SystemExit
        If the provided model_id is not an integer equal to 0 or 1.

    Notes
    -----
    This function filters the chain in-place, meaning that the original chain will be modified.
    """

    nmodel_idx = list(params).index('nmodel')

    if model_id is None:
        print("No model ID specified, posteriors are plotted for model 0")
        filter_model = chain[:, nmodel_idx] < 0.5
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
    filtered_par = [par.replace('np-', '') if 'gw-bhb-np' in par else par for par in filtered_par]

    return chain[:, filter_par], filtered_par


def calc_df(chain,):
    """
    Input
    ============
    chain: input dropout chain, with shape assumed to be n_bootstrap x n_samples

    Output
    ============
    bayes_facs: dropout bayes factor, with shape n_bootstrap
    """

    bayes_facs = np.full(chain.shape[0], 0.0)

    for ii, arr in enumerate(chain):
        bayes_facs[ii] = model_utils.odds_ratio(arr)[0]

    return bayes_facs


def bf_bootstrap(chain, burn=0):
    """
    This function computes the mean and variance of the bayes factor after bootstrapping 
    for a given chain.

    Parameters
    ----------
    chain : numpy.ndarray
        The Markov Chain Monte Carlo (MCMC) chain to be analyzed. This should be a multi-dimensional array where each row 
        represents a state in the chain, and each column represents a parameter.

    burn : int, optional
        The burn-in period to be discarded from the start of the chain. This should be a non-negative integer. 
        If not provided, no burn-in period will be discarded.

    Returns
    -------
    tuple
        A tuple containing the mean and variance of the bootstrapped bayes factors:

        - mean: float
            The mean of the bootstrapped degrees of freedom distribution.

        - var: float
            The variance of the bootstrapped degrees of freedom distribution.

    Notes
    -----
    This function uses the 'acor' library to compute the autocorrelation time of the chain, which is then used to thin the chain. 
    The thinned chain is then bootstrapped using the 'bootstrap' function with the 'calc_df' user statistic, to obtain a distribution 
    of degrees of freedom. The mean and variance of this distribution are then computed.
    """

    corr_len = int(acor.acor(chain[burn:,-1])[0])

    test = chain[burn::corr_len]

    df_dist_bs = bootstrap(test[:,0], n_bootstraps = 50000, user_statistic = calc_df)

    mean = np.mean(df_dist_bs)
    var = np.var(df_dist_bs)

    return mean, var


def compute_bf(chain, params, bootstrap=False):
    """
    Computes the Bayes factor and estimate its uncertainty.

    Parameters
    ----------
    chain : numpy.ndarray
        The Markov Chain Monte Carlo (MCMC) chain to be analyzed. This should be a multi-dimensional array where each row represents 
        a state in the chain, and each column represents a parameter. The 'nmodel' and 'log_posterior' columns should be used to specify the model index and the log of the posterior probabilities.

    params : list of str
        The names of the parameters in the chain. This should be a list of strings with the same length as the number of columns in the chain. 
        It is expected to contain 'nmodel' and 'log_posterior', which will be used to filter the chain based on the model index and compute the Bayes factor.

    bootstrap : bool, optional
        A flag indicating whether to compute the Bayes factor using a bootstrap method. If True, the Bayes factor will be computed 
        using the 'get_bf' function. The bootsrap calculation is significantly slower.
        Defaults to False.

    Returns
    -------
    tuple
        A tuple containing the following elements:

        - bf: float
            The computed Bayes factor. This gives the evidence for model 0 over model 1. A higher value provides stronger evidence for model 0.
        
        - unc: float
            The computed uncertainty of the Bayes factor.

    Raises
    ------
    ValueError
        If 'nmodel' or 'log_posterior' is not found in the 'params' list.

    """

    nmodel_idx = list(params).index('nmodel')
    posterior_idx = list(params).index('log_posterior')

    if bootstrap:
        data = chain[:,[nmodel_idx, posterior_idx]]
        bf, unc = bf_bootstrap(data)

    else:
        bf, unc = model_utils.odds_ratio(chain[:, nmodel_idx], models=[0,1])

    return bf, unc


def bisection(f, a, b, tol): 
    """
    This function implements the bisection method for root finding of a real-valued function. It recursively divides 
    the interval [a, b] into two subintervals until the absolute value of f evaluated at the midpoint is less than 
    the specified tolerance, at which point it returns the midpoint as an approximation of the root.

    Parameters
    ----------
    f : function
        The function for which the root is to be found. It must be real-valued and continuous on the interval [a, b].

    a : float
        The left endpoint of the interval in which the root is sought. It must be less than b.

    b : float
        The right endpoint of the interval in which the root is sought. It must be greater than a.

    tol : float
        The tolerance for the root approximation. The function will return when the absolute value of f evaluated at 
        the midpoint is less than tol. It must be greater than 0.

    Returns
    -------
    float or None
        The midpoint of the final subinterval if a root is found; None otherwise. The root approximation m is guaranteed 
        to satisfy |f(m)| < tol if the function converges.

    Raises
    ------
    ValueError
        If a is not less than b, or if tol is not greater than 0.

    Notes
    -----
    This is a recursive implementation of the bisection method. The bisection method assumes that the function f changes 
    sign over the interval [a, b], which implies that a root exists in this interval by the Intermediate Value Theorem.
    """
    
    if np.sign(f(a)) == np.sign(f(b)):
        return None
        
    m = (a + b)/2
    
    if np.abs(f(m)) < tol:
        return m
    elif np.sign(f(a)) == np.sign(f(m)):
        return bisection(f, m, b, tol)
    elif np.sign(f(b)) == np.sign(f(m)):
        return bisection(f, a, m, tol)


def k_ratio_aux_1D(
    sample, 
    bf, 
    par,
    par_range,
    k_ratio):
    """
    Returns the bound value for a given k-ratio in a 1D posterior density plot.

    Parameters
    ----------
    sample : MCSamples
        An instance of the MCSamples class, containing the multivariate 
        Monte Carlo samples on which the function is operating.
    bf : float
        The Bayes factor for the exotic + SMBHB vs. SMBHB model. 
        Represents the strength of evidence in favour of the exotic model.
    par : str
        The name of the parameter for which the k-ratio bound should be computed.
    par_range : list of float
        The lower and upper prior limits for the parameter. It is represented as a list where 
        the first element is the lower limit and the second element is the upper limit.
    k_ratio : float
        The fraction of plateau height at which the height level is determined. This is used 
        to compute the height_KB, which represents the height at which the bound is computed.

    Returns
    -------
    float
        The computed k-ratio bound value. This is the value of the parameter at which the 
        1D posterior density crosses the height_KB. 

    Raises
    ------
    ValueError
        If the integration or the bisection search fails due to numerical issues or 
        if the specified parameter range does not contain a valid root.
    """
    
    density1D = MCSamples.get1DDensity(sample, par)
    
    norm = integrate.quad(density1D, par_range[0], par_range[1], full_output=1)[0]
    
    prior = 1/(par_range[1]-par_range[0])
    
    height_KB = k_ratio*prior/bf*norm
    
    k_val = bisection(f = (lambda x: density1D(x)-height_KB), a = par_range[0], b = par_range[1], tol = 10**(-8))

    return k_val


def k_ratio_aux_2D(
    sample, 
    bf, 
    par_1, 
    par_2, 
    par_range_1, 
    par_range_2, 
    k_ratio):
    """
    Returns the height level corresponding to the given k-ratio in a 2D posterior density plot.

    Parameters
    ----------
    sample : MCSamples
        An instance of the MCSamples class, containing the multivariate 
        Monte Carlo samples on which the function is operating.
    bf : float
        The Bayes factor for the exotic + SMBHB vs. SMBHB model. 
        Represents the strength of evidence in favour of the exotic model.
    par_1, par_2 : str
        The names of the two parameters for which the k-ratio bound should be computed.
    par_range_1, par_range_2 : list of float
        The lower and upper prior limits for the parameters. Each is represented as a list 
        where the first element is the lower limit and the second element is the upper limit.
    k_ratio : float
        The fraction of plateau height at which the height level is determined. This is used 
        to compute the height_KB, which represents the height at which the bound is computed.

    Returns
    -------
    float
        The computed height level in the 2D posterior density plot. This is the height at which 
        the density equals the computed height_KB.

    Raises
    ------
    ValueError
        If the double integration fails due to numerical issues.
    """
    density2D = MCSamples.get2DDensity(sample, par_1, par_2)
    
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        norm = integrate.dblquad(density2D, par_range_2[0], par_range_2[1], par_range_1[0], par_range_1[1])[0]#Calculate prior value at each point
    
    prior = 1/(par_range_2[1]-par_range_2[0]) * 1/(par_range_1[1]-par_range_1[0])
    
    height_KB = k_ratio*prior/bf*norm

    return height_KB


def get_k_levels(sample, pars, priors, bf, k_ratio):
    """
    Computes and returns the 1D and 2D k-ratio bounds for a given set of parameters.

    Parameters
    ----------
    sample : MCSamples
        An instance of the MCSamples class, containing the multivariate 
        Monte Carlo samples on which the function is operating.
    pars : list of str
        The list of all parameters for which the k-ratio bounds should be computed. 
        The parameters 'gw-bhb-0' and 'gw-bhb-1' are excluded from this computation.
    priors : dict
        A dictionary containing the lower and upper prior limits for each parameter. 
        Each key-value pair in the dictionary corresponds to a parameter and its limits, 
        respectively.
    bf : float
        The Bayes factor for the exotic + SMBHB vs. SMBHB model. 
        Represents the strength of evidence in favour of the exotic model.
    k_ratio : float
        The fraction of plateau height at which the height level is determined.

    Returns
    -------
    np.ndarray, np.ndarray
        Two numpy arrays representing the 1D and 2D k-ratio bounds, respectively. 
        Each element in the arrays is a list where the first elements are the parameter names 
        and the last element is the computed k-ratio bound.
    """

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



def get_bayes_est(samples, params):
    """
    Computes and returns the Bayesian estimates for a given set of parameters 
    based on a sample of data.

    Parameters
    ----------
    samples : MCSamples
        An instance of the MCSamples class, containing the multivariate 
        Monte Carlo samples on which the function is operating.
    params : list of str
        The list of parameters for which the Bayesian estimates should be computed.

    Returns
    -------
    dict
        A dictionary representing the Bayesian estimates for each parameter. 
        Each key-value pair in the dictionary corresponds to a parameter and its 
        Bayesian estimate, respectively. Each estimate is represented as a tuple, 
        where the first element is the mean and the second element is the standard 
        deviation.
    """
    x = list(samples.setMeans())
    xerr = list(np.sqrt(samples.getVars()))
    x = zip(x, xerr)
    x = dict(zip(params, x))
    return x


def get_max_pos(params, bayes_est, sample, priors, spc=10):
    """
    Computes and returns the maximum posterior position for a given set of parameters.

    Parameters
    ----------
    params : list of str
        The list of parameters for which the maximum posterior position should be computed.
    bayes_est : dict
        A dictionary containing the Bayesian estimates for each parameter. 
        Each key-value pair in the dictionary corresponds to a parameter and its 
        Bayesian estimate, respectively. Each estimate is represented as a tuple, 
        where the first element is the mean and the second element is the standard 
        deviation.
    sample : MCSamples
        An instance of the MCSamples class, containing the multivariate 
        Monte Carlo samples on which the function is operating.
    priors : dict
        A dictionary containing the lower and upper prior limits for each parameter. 
        Each key-value pair in the dictionary corresponds to a parameter and its limits, 
        respectively.
    spc : int, optional
        The number of equally spaced points to be considered within the bounds of 
        each parameter when searching for the maximum posterior position. Default is 10.

    Returns
    -------
    dict
        A dictionary representing the maximum posterior positions for each parameter. 
        Each key-value pair in the dictionary corresponds to a parameter and its 
        maximum posterior position, respectively.
    """

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


def get_c_levels(sample, pars, levels):
    """
    Computes and returns the highest posterior interval (HPI) for a given set of parameters 
    and confidence levels.

    Parameters
    ----------
    sample : MCSamples
        An instance of the MCSamples class, containing the multivariate 
        Monte Carlo samples on which the function is operating.
    pars : list of str
        The list of parameters for which the HPI should be computed.
    levels : list of float
        The list of confidence levels for which the HPI should be computed. 
        Each value in the list should be between 0 and 1.

    Returns
    -------
    np.ndarray
        A numpy array representing the HPI for each parameter and each confidence level. 
        Each element in the array is a list, where the first element is a parameter name and 
        the second element is a list of HPIs for each confidence level.
    """
    
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
