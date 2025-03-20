"""Utilities to be used with output MCMC chains of PTArcade."""
from __future__ import annotations

import logging
import os
import re
import time
import warnings
from collections.abc import Callable
from datetime import datetime as dt
from itertools import combinations
from pathlib import Path
from typing import Any

import acor
import numpy as np
import pandas as pd
from astroML.resample import bootstrap
from enterprise_extensions import model_utils
from getdist.mcsamples import MCSamples
from numpy.typing import NDArray
from scipy import integrate
from scipy.optimize import Bounds, minimize

log = logging.getLogger("rich")

def params_loader(file: str | Path) -> dict[str, tuple[float, float] | None]:
    """Load a parameter file and return a dictionary of the parameters with their respective values or None.

    Parameters
    ----------
    file : str | Path
        The path to the prior.txt file.

    Returns
    -------
    params : dict[str, tuple[float, float] | None]
        A dictionary with the parameters' prior ranges.
    """
    params = {}

    with open(file) as f:
        for line in f:
            if ':' not in line:
                continue

            key = line.split(":")[0]

            if "Uniform" in line:
                minimum = float(re.search('pmin=(.*?),', line).group(1)) # type: ignore
                maximum = float(re.search('pmax=(.*?)\\)', line).group(1)) # type: ignore

                params[key] = (minimum, maximum)

            elif "Normal" in line:
                if '[' in line:
                    dim = len(re.search('\\[(.*?)\\]', line).group(1).split()) # type: ignore
                else:
                    dim = 1
                for i in range(dim):
                    params[f"{key}_{i}"] = None # type: ignore

            else:
                params[key] = None # type: ignore

    return params # type: ignore

def convert_chains_to_hdf(
    chains_dir: str | Path,
    burn_frac: float = 0.0,
    quick_import: bool = False,  # noqa: FBT001, FBT002
    chain_name: str = "chain_1.txt",
    dest_path: Path | None = None,
    **kwargs,
) -> None:
    """Convert the raw output of PTArcade to HDF format and write to disk.

    Parameters
    ----------
    chains_dir : str | Path
        Name of the directory containing the chains.
    burn_frac : float, optional
        Fraction of the chain that is removed from the head (default is 0).
    quick_import : bool, optional
        Flag to skip importing the rednoise portion of chains (default is False).
    chain_name : str, optional
        The name of the chain files, include the file extension of the chain files.
        Compressed files with extension ".gz" can be used (default is "chain_1.txt").
    dest_path : Path | None
        The destination path including filename (default is to save
        in chains_dir with a unique timestamp).
    **kwargs : dict
        Additional arguments passed to [pandas.DataFrame.to_hdf][]

    Returns
    -------
    None
        Saves an HDF5 file to `dest_path` in "table" format. There is one group for each
        chain ("chain_N"), with a "table" dataset in each group that contains the
        MCMC samples. There is also a single group for parameters called "parameters",
        also with a dataset called "table".

    """
    if isinstance(chains_dir, str):
        chains_dir = Path(chains_dir)

    use_index = kwargs.pop("index", False)
    complevel = kwargs.pop("complevel", 9)
    complib = kwargs.pop("complib", "blosc:lz4")

    if dest_path is None:
        dest_path = (
            chains_dir / f"chains-{chains_dir.name}-{dt.now().strftime('%Y%m%d_%H%M%S')}").with_suffix( # noqa: DTZ005
            ".h5",
        )

    params, dataframes = import_to_dataframe(chains_dir, burn_frac, quick_import, chain_name, merge_chains=False)

    for num, frame in enumerate(dataframes):
        frame.to_hdf(dest_path, key=f"chain_{num}", index=use_index, complevel=complevel, complib=complib, **kwargs)

    params.to_hdf(dest_path, key="parameters", index=use_index, complevel=complevel, complib=complib, **kwargs)
    print(f"Successfully converted to HDF. Saved at {dest_path}.")


def import_to_dataframe(
    chains_dir: str | Path,
    burn_frac: float = 0.0,
    quick_import: bool = False,  # noqa: FBT001, FBT002
    chain_name: str = "chain_1.txt",
    merge_chains: bool = True, # noqa: FBT001, FBT002
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Import the chains and their parameter file as a pandas dataframe.

    Given a `chain_dir` that contains chains, this function imports the chains, removes `burn_frac`, and returns
    a dictionary of parameters: values and a numpy array of the merged chains.

    Parameters
    ----------
    chains_dir : str | Path
        Name of the directory containing the chains.
    burn_frac : float, optional
        Fraction of the chain that is removed from the head (default is 0).
    quick_import : bool, optional
        Flag to skip importing the rednoise portion of chains (default is False).
    chain_name : str, optional
        The name of the chain files, include the file extension of the chain files. Compressed files
        with extension ".gz" and HDF5 files with extension ".h5" can be used (default is "chain_1.txt").
    merge_chains : bool, optional
        Whether to merge the chains into one dataframe (default is True).

    Returns
    -------
    params : pandas.DataFrame
        Dataframe containing the parameter names and their values.
    chains : pandas.DataFrame | list[pandas.DataFrame]
        Dataframe array containing the merged chains without the burn-in region. Can also optionally
        return a list of unmerged chains if `merg_chains` is False.

    Raises
    ------
    SystemExit
        Raised when the chains have different parameters.

    """
    print("Starting import from", chains_dir)

    start_time = time.time()

    if isinstance(chains_dir, str):
        chains_dir = Path(chains_dir)

    chain_ext = Path(chain_name).suffix

    if chain_ext == ".txt" or chain_ext == ".gz":
        # get all the chain directories
        directories = [x for x in chains_dir.iterdir()
                       if x.is_dir()]

        # get the parameter lits and check that all the chains have the same parameters
        params = params_loader(directories[0] / "priors.txt")

        for chain in directories:
            temp_par = params_loader(chain / "priors.txt")

            if temp_par != params:
                err = f"Chains have different parameters. {temp_par=} != {params=}"
                log.error(err)
                raise SystemExit

        n_pars = len(np.loadtxt(directories[0] / "chain_1.txt", max_rows=1))

        # add name for sampler parameters
        if n_pars - len(params) == 5:
            params.update(
                {
                    "nmodel": None,
                    "log_posterior": None,
                    "log_likelihood": None,
                    "acceptance_rate": None,
                    "n_parall": None,
                },
            )
        elif n_pars - len(params) == 4:
            params.update({"log_posterior": None, "log_likelihood": None, "acceptance_rate": None, "n_parall": None})

        # import and merge all the chains removing the burn-in
        name_list = list(params.keys())

        red_noise_filter = re.compile(".*red_noise.*")

        if quick_import and any(filter(red_noise_filter.match, name_list)): # type: ignore
            # Search reversed list for first occurence of "red_noise"
            # Return the index (remember, the list is reversed!)
            # The use of `next` and a generator makes it so that we don't have to
            # search the whole list, we stop when we get the first match
            red_noise_ind = next(i for i in enumerate(name_list[::-1]) if "red_noise" in i[1])[0]

            # Slice the list so that we begin directly after the index found above
            usecols = name_list[-1 * red_noise_ind :]

            params = {name: params[name] for name in usecols}
        else:
            usecols = name_list
        dtypes = {name: float for name in usecols}
        # import and merge all the chains removing the burn-in


        if merge_chains:
            chains = pd.concat(
                (
                    pd.read_csv(
                        (drct / chain_name),
                        sep="\t",
                        names=name_list,
                        dtype=dtypes,
                        usecols=usecols,
                    ).iloc[lambda x: int(len(x) * burn_frac) <= x.index]
                    for drct in directories
                ),
                ignore_index=True,
                sort=False,
            )

            chains = chains.dropna()

        else:
            chains = [
                pd.read_csv(
                    (drct / chain_name),
                    sep="\t",
                    names=name_list,
                    dtype=dtypes,
                    usecols=usecols,
                ).iloc[lambda x: int(len(x) * burn_frac) <= x.index]
                for drct in directories
            ]

        params = pd.DataFrame.from_dict(params).fillna(np.nan)  # need to fill `None` with a numeric value

    elif chain_ext == ".h5":

        with pd.HDFStore(chains_dir / chain_name) as h5:
            keys = h5.keys()
            chain_keys = [key for key in keys if "chain" in key]

            params = h5.get("/parameters")

            chains = pd.concat(
                [
                    pd.read_hdf(h5, key=chain_key).iloc[lambda x: int(len(x) * burn_frac) <= x.index]
                    for chain_key in chain_keys
                ],
                ignore_index=True,
                sort=False,
            )

            chains = chains.dropna()


    print(f"Finished importing   {chains_dir} in {time.time() - start_time:.2f}s")

    return params, chains

def import_chains(chains_dir: str | Path,
                  burn_frac: float = 1/4,
                  quick_import: bool = True, # noqa: FBT001, FBT002
                  chain_name: str = "chain_1.txt") -> tuple[dict, NDArray]:
    """Import the chains and their parameter file.

    Given a `chain_dir` that contains chains, this function imports the chains, removes `burn_frac`, and returns
    a dictionary of parameters: values and a numpy array of the merged chains.

    Parameters
    ----------
    chains_dir : str | Path
        Name of the directory containing the chains.
    burn_frac : float, optional
        Fraction of the chain that is removed from the head (default is 1/4).
    quick_import : bool, optional
        Flag to skip importing the rednoise portion of chains (default is True).
    chain_name : str, optional
        The name of the chain files, include the file extension of the chain files. Compressed files
        with extension ".gz" and HDF5 files with extension ".h5" can be used (default is "chain_1.txt").

    Returns
    -------
    params : dict
        Dictionary containing the parameter names and their values.
    mrgd_chain : NDArray
        Numpy array containing the merged chains without the burn-in region.

    Raises
    ------
    SystemExit
        Raised when the chains have different parameters.

    """
    params, mrgd_chain = import_to_dataframe(chains_dir, burn_frac, quick_import, chain_name)

    mrgd_chain = mrgd_chain.to_numpy(dtype=float)
    params = params.to_dict(orient="list")

    return params, mrgd_chain


def chain_filter(
        chain: NDArray,
        params: list[str],
        model_id: int | None,
        par_to_plot: list[str] | None,
) -> tuple[NDArray, list[str]]:
    """Filter chains.

    This function filters the rows in the provided chain according to the specified model and parameters. It selects
    rows that correspond to the specified model ID and parameters to plot their posteriors.

    Parameters
    ----------
    chain : NDArray
        The Markov Chain Monte Carlo (MCMC) chain to be filtered. This should be a multi-dimensional array where each
        row represents a state in the chain, and each column represents a parameter.
    params : list[str]
        The names of the parameters in the chain. This should be a list of strings with the same length as the number
        of columns in the chain.
    model_id : int | None
        The ID of the model to filter the chain for. This should be either 0 or 1. If None, the function will select
        rows for model 0.
    par_to_plot : list[str] | None
        The names of the parameters to filter the chain for. If None, the function will select all parameters except
        'nmodel', 'log_posterior', 'log_likelihood', 'acceptance_rate', and 'n_parall', and parameters
        containing '+' or '-'.

    Returns
    -------
    chain : NDArray
        The filtered chain, containing only rows corresponding to the specified model ID and parameters.
    filtered_par: list[str]
        The list of filtered parameter names.

    Raises
    ------
    SystemExit
        If the provided `model_id` is not an integer equal to 0 or 1.

    Notes
    -----
    This function filters the chain in-place, meaning that the original chain will be modified.

    """
    if 'nmodel' in list(params):
        nmodel_idx = list(params).index('nmodel')

        if model_id is None:
            print("No model ID specified, posteriors are plotted for model 0")
            filter_model = chain[:, nmodel_idx] < 0.5
        elif model_id == 0:
            filter_model = chain[:, nmodel_idx] < 0.5
        elif model_id == 1:
            filter_model = chain[:, nmodel_idx] > 0.5
        else:
            err = f"'model_idx' can only be an integer equal to 0 or 1. I got {model_id=}."
            log.error(err)
            raise SystemExit

        chain = chain[filter_model]
    else:
        pass

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

    filtered_par = [par.replace('_','-') for par in params[filter_par]] # type: ignore
    filtered_par = [par.replace('np-', '') if 'gw-bhb-np' in par else par for par in filtered_par]

    return chain[:, filter_par], filtered_par


def calc_df(chain: NDArray) -> NDArray:
    """Calculate dropout bayes factor

    Parameters
    ----------
    chain : NDArray
        input dropout chain with shape assumed to be (n_bootstrap, n_samples)

    Returns
    -------
    bayes_facs : NDArray
        dropout bayes factor with shape (n_bootstrap,)
    """
    bayes_facs = np.full(chain.shape[0], 0.0)

    for ii, arr in enumerate(chain):
        bayes_facs[ii] = model_utils.odds_ratio(arr)[0]

    return bayes_facs


def bf_bootstrap(chain: NDArray, burn: int = 0) -> tuple[float, float]:
    """Compute mean and variance of bayes factor.

    This function computes the mean and variance of the bayes factor after bootstrapping
    for a given chain.

    Parameters
    ----------
    chain : NDArray
        The Markov Chain Monte Carlo (MCMC) chain to be analyzed. This should be a multi-dimensional array where each
        row represents a state in the chain, and each column represents a parameter.
    burn : int, optional
        The burn-in period to be discarded from the start of the chain. This should be a non-negative integer.
        If not provided, no burn-in period will be discarded.

    Returns
    -------
    mean : float
        The mean of the bootstrapped degrees of freedom distribution.
    var : float
        The variance of the bootstrapped degrees of freedom distribution.

    Notes
    -----
    This function uses the `acor` library to compute the autocorrelation time of the chain, which is then used to thin
    the chain. The thinned chain is then bootstrapped using the 'bootstrap' function with the `calc_df` user statistic,
    to obtain a distribution of degrees of freedom. The mean and variance of this distribution are then computed.

    """
    corr_len = int(acor.acor(chain[burn:,-1])[0])

    test = chain[burn::corr_len]

    df_dist_bs = bootstrap(test[:,0], n_bootstraps = 50000, user_statistic = calc_df)

    mean = np.mean(df_dist_bs)
    var = np.var(df_dist_bs)

    return mean, np.sqrt(var)


def compute_bf(chain: NDArray, params: list[str], bootstrap: bool = False) -> tuple[float, float]:  # noqa: FBT001, FBT002
    """Compute the Bayes factor and estimate its uncertainty.

    Parameters
    ----------
    chain : NDArray
        The Markov Chain Monte Carlo (MCMC) chain to be analyzed. This should be a multi-dimensional array where each
        row represents a state in the chain, and each column represents a parameter. The 'nmodel' and 'log_posterior'
        columns should be used to specify the model index and the log of the posterior probabilities.
    params : list[str]
        The names of the parameters in the chain. This should be a list of strings with the same length as the number of
        columns in the chain. It is expected to contain 'nmodel' and 'log_posterior', which will be used to filter the
        chain based on the model index and compute the Bayes factor.
    bootstrap : bool, optional
        A flag indicating whether to compute the Bayes factor using a bootstrap method. If True, the Bayes factor will
        be computed using the 'get_bf' function. The bootsrap calculation is significantly slower. Defaults to False.

    Returns
    -------
    bf : float
        The computed Bayes factor. This gives the evidence for model 1 over model 0. A higher value provides
        stronger evidence for model 1.
    unc : float
        The computed uncertainty of the Bayes factor.

    Raises
    ------
    SystemExit
        If `nmodel` or `log_posterior` is not found in the `params` list.

    """
    try:
        nmodel_idx = list(params).index('nmodel')
    except ValueError:
        err = ("'nmodel' was not found in params.\n"
               f"You supplied {params=}")
        log.error(err)
        raise SystemExit from None
    try:
        posterior_idx = list(params).index('log_posterior')
    except ValueError:
        err = ("'log_posterior' was not found in params.\n"
               f"You supplied {params=}")
        log.error(err)
        raise SystemExit from None


    if bootstrap:
        data = chain[:,[nmodel_idx, posterior_idx]]
        bf, unc = bf_bootstrap(data)

    else:
        bf, unc = model_utils.odds_ratio(chain[:, nmodel_idx], models=[0,1])

    return bf, unc


def bisection(f: Callable[[Any], float], a: float, b: float, tol: float) -> float | None:
    """Find roots for real-valued function using bisection method.

    This function implements the bisection method for root finding of a real-valued function. It recursively divides
    the interval [a, b] into two subintervals until the absolute value of f evaluated at the midpoint is less than
    the specified tolerance, at which point it returns the midpoint as an approximation of the root.

    Parameters
    ----------
    f : Callable[[Any], float]
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
    float | None
        The midpoint of the final subinterval if a root is found; None otherwise. The root approximation m is guaranteed
        to satisfy |f(m)| < tol if the function converges.

    Raises
    ------
    SystemExit
        If a is not less than b, or if tol is not greater than 0.

    Notes
    -----
    This is a recursive implementation of the bisection method. The bisection method assumes that the function f changes
    sign over the interval [a, b], which implies that a root exists in this interval by the Intermediate Value Theorem.

    """
    if a >= b:
        err = f"a is not less than b {a=} {b=}."
        log.error(err)
        raise SystemExit

    if tol <= 0:
        err = f"tol is not greater than 0 {tol=}."
        log.error(err)
        raise SystemExit

    if np.sign(f(a)) == np.sign(f(b)):
        return None

    m = (a + b) / 2

    if np.abs(f(m)) < tol:
        return m
    elif np.sign(f(a)) == np.sign(f(m)):
        return bisection(f, m, b, tol)
    elif np.sign(f(b)) == np.sign(f(m)):
        return bisection(f, a, m, tol)
    else:
        return None


def k_ratio_aux_1D(
    sample: MCSamples,
    bf: float,
    par: str,
    par_range: list[float],
    k_ratio: float) -> float | None:
    """Returns the bound value for a given k-ratio in a 1D posterior density plot.

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
    par_range : list[float]
        The lower and upper prior limits for the parameter. It is represented as a list where
        the first element is the lower limit and the second element is the upper limit.
    k_ratio : float
        The fraction of plateau height at which the height level is determined. This is used
        to compute the height_KB, which represents the height at which the bound is computed.

    Returns
    -------
    k_val : float
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
    sample: MCSamples,
    bf: float,
    par_1: str,
    par_2: str,
    par_range_1: list[float],
    par_range_2: list[float],
    k_ratio: float) -> float:
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
    par_range_1, par_range_2 : list[float]
        The lower and upper prior limits for the parameters. Each is represented as a list
        where the first element is the lower limit and the second element is the upper limit.
    k_ratio : float
        The fraction of plateau height at which the height level is determined. This is used
        to compute the height_KB, which represents the height at which the bound is computed.

    Returns
    -------
    height_KB : float
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
        #Calculate prior value at each point
        norm = integrate.dblquad(density2D, par_range_2[0], par_range_2[1], par_range_1[0], par_range_1[1])[0]

    prior = 1/(par_range_2[1]-par_range_2[0]) * 1/(par_range_1[1]-par_range_1[0])

    height_KB = k_ratio*prior/bf*norm

    return height_KB


def get_k_levels(sample: MCSamples,
                 pars: list[str],
                 priors: dict,
                 bf: float,
                 k_ratio: float) -> tuple[NDArray, NDArray]:
    """Compute and return the 1D and 2D k-ratio bounds for a given set of parameters.

    Parameters
    ----------
    sample : MCSamples
        An instance of the MCSamples class, containing the multivariate
        Monte Carlo samples on which the function is operating.
    pars : list[str]
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
    NDArray
        numpy array representing the 1D k-ratio bounds.
        Each element in the array is a list where the first elements are the parameter names
        and the last element is the computed k-ratio bound.
    NDArray
        numpy array representing the 2D k-ratio bounds.
        Each element in the array is a list where the first elements are the parameter names
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
                bf=bf,
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
                bf = bf,
                par = par,
                par_range = priors[par],
                k_ratio = k_ratio)])

    return np.array(levels_1d), np.array(levels_2d)



def get_bayes_est(samples: MCSamples, params: list[str]) -> dict[str, tuple[float, float]]:
    """Compute and return the Bayesian estimates for a given set of parameters based on a sample of data.

    Parameters
    ----------
    samples : MCSamples
        An instance of the MCSamples class, containing the multivariate
        Monte Carlo samples on which the function is operating.
    params : list[str]
        The list of parameters for which the Bayesian estimates should be computed.

    Returns
    -------
    x : dict[str, tuple[float, float]]
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


def get_max_pos(params: list[str],
                bayes_est: dict[str, tuple[float, float]],
                sample: MCSamples,
                priors: dict[str, tuple[float, float]],
                spc: int = 10) -> dict[str, float]:
    """Compute and return the maximum posterior position for a given set of parameters.

    Parameters
    ----------
    params : list[str]
        The list of parameters for which the maximum posterior position should be computed.
    bayes_est : dict[str, tuple[float, float]]
        A dictionary containing the Bayesian estimates for each parameter.
        Each key-value pair in the dictionary corresponds to a parameter and its
        Bayesian estimate, respectively. Each estimate is represented as a tuple,
        where the first element is the mean and the second element is the standard
        deviation.
    sample : MCSamples
        An instance of the MCSamples class, containing the multivariate
        Monte Carlo samples on which the function is operating.
    priors : dict[str, tuple[float, float]]
        A dictionary containing the lower and upper prior limits for each parameter.
        Each key-value pair in the dictionary corresponds to a parameter and its limits,
        respectively.
    spc : int, optional
        The number of equally spaced points to be considered within the bounds of
        each parameter when searching for the maximum posterior position. Default is 10.

    Returns
    -------
    out : dict[str, float]
        A dictionary representing the maximum posterior positions for each parameter.
        Each key-value pair in the dictionary corresponds to a parameter and its
        maximum posterior position, respectively.

    """
    out = {}

    for par in params:
        bounds = priors.get(par)

        density =  MCSamples.get1DDensity(sample, par, normalized=False)

        mind = lambda x: -density(x)
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


def get_c_levels(sample: MCSamples, pars: list[str], levels: list[float]) -> NDArray:
    """Compute and return the highest posterior interval (HPI) for a given set of parameters and confidence levels.

    Parameters
    ----------
    sample : MCSamples
        An instance of the MCSamples class, containing the multivariate
        Monte Carlo samples on which the function is operating.
    pars : list[str]
        The list of parameters for which the HPI should be computed.
    levels : list[float]
        The list of confidence levels for which the HPI should be computed.
        Each value in the list should be between 0 and 1.

    Returns
    -------
    NDArray
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
