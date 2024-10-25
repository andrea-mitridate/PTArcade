"""Import PTA data."""
from __future__ import annotations

import glob
import json
import logging
import os
import pickle

from astropy.utils.data import download_file, get_readable_fileobj
from enterprise.pulsar import Pulsar
from numpy._typing import _ArrayLikeFloat_co as array_like

log = logging.getLogger("rich")

def get_ephem_conv(par_file: str) -> str:
    """Get the ephemeris convention used in par files.

    Parameters
    ----------
    par_file : str
        Path to the par file.

    Returns
    -------
    ephem : str
        The ephemeris convention used in the par file.

    """
    f = open(par_file)
    lines = f.readlines()

    for line in lines:
        line = line.split() # type: ignore
        if line[0] == "EPHEM":
            ephem = line[-1]

    f.close()

    return ephem


def get_pulsars(pta_data: str, filters: list[str] | None = None) -> list[Pulsar]:
    """Get Pulsar data.

    If `pta_data` is a file, attempt to load it as a pickle. If it is a
    directory, read in the par and tim files within.

    Parameters
    ----------
    pta_data : str
        Filename for pickle or directory containing par and tim files.
    filters : list[str]
        Selective filter for par and tim files.

    Returns
    -------
    psrs : list[Pulsar]
        List containing pulsar data from `pta_data`.

    Raises
    ------
    SystemExit
        If `pta_data` is not a file or directory.

    """
    if os.path.isfile(pta_data):
        with get_readable_fileobj(pta_data, "binary") as handle:
            psrs = pickle.load(handle)

        return psrs

    elif os.path.isdir(pta_data):
        parfiles = sorted(glob.glob(os.path.join(pta_data, "*.par")))
        timfiles = sorted(glob.glob(os.path.join(pta_data, "*.tim")))

        # filter
        if filters is not None:
            parfiles = [x for x in parfiles if x.split("/")[-1].split(".")[0] in filters]
            timfiles = [x for x in timfiles if x.split("/")[-1].split(".")[0] in filters]
        # TODO Need to check that all tim and par files have matches!

        # load the pulsars into enterprise
        psrs = []
        for p, t in zip(parfiles, timfiles):
            ephemeris = get_ephem_conv(p)
            psr = Pulsar(p, t, ephem=ephemeris)
            psrs.append(psr)

        return psrs
    else:
        err = f"'pta_data' is not correct. Must be path or file.\nCurrent value is {pta_data=}."
        log.error(err)
        raise SystemExit


def get_wn(wn_data: str | None) -> dict | None :
    """Get whitenoise data.

    Parameters
    ----------
    wn_data : str, optional
        File or directory containing whitenoise data.

    Returns
    -------
    dict | None
        If `wn_data` isn't `None`, then the whitenoise data is returned as a
        `dict`

    """
    params = {}
    if wn_data is None:
        return None

    elif os.path.isfile(wn_data):
        with open(wn_data) as fp:
            params.update(json.load(fp))

        return params

    elif os.path.isdir(wn_data):
        for filename in os.listdir(wn_data):
            with open(os.path.join(wn_data, filename)) as f:
                data = json.load(f)
                for key, value in data.items():
                    params.update({key: value})

        return params

    else:
        err = f"'wn_data' is not correct. Must be path, file, or None.\nCurrent value is {wn_data=}."
        log.error(err)
        raise SystemExit


def get_rn(rn_data:str | None) -> dict | None:
    """
    Get intrinsic red (spin) noise frequencies

    Parameters
    ----------
    rn_data : str, optional
        File or directory containing intrinsic red noise data.

    Returns    
    -------
    dict | None
        If `rn_data` isn't `None`, then the red noise data is returned as a
        `dict`
    """

    red_dict = {}
    if rn_data is None:
        return None
    elif os.path.isfile(rn_data):
        with open(rn_data) as fp:
            red_dict.update(json.load(fp))
        return red_dict

    else:
        err = f"'rn_data' is not correct. Must be file or None.\nCurrent value is {rn_data=}."
        log.error(err)
        raise SystemExit
    

def get_cn(cn_data:str | None) -> dict | None:
    """
    Get chromatic noise frequencies

    Parameters
    ----------
    cn_data : str, optional
        File or directory containing chromatic noise data.

    Returns    
    -------
    dict | None
        If `cn_data` isn't `None`, then the chromatic noise data is returned as a
        `dict`
    """

    chrom_dict = {}
    if cn_data is None:
        return None
    elif os.path.isfile(cn_data):
        with open(cn_data) as fp:
            chrom_dict.update(json.load(fp))
        return chrom_dict

    else:
        err = f"'cn_data' is not correct. Must be file or None.\nCurrent value is {cn_data=}."
        log.error(err)
        raise SystemExit
    

def get_dmn(dmn_data:str | None) -> dict | None:
    """
    Get dm noise frequencies

    Parameters
    ----------
    dmn_data : str, optional
        File or directory containing dm noise data.

    Returns    
    -------
    dict | None
        If `dmn_data` isn't `None`, then the dm noise data is returned as a
        `dict`
    """

    dm_dict = {}
    if dmn_data is None:
        return None
    elif os.path.isfile(dmn_data):
        with open(dmn_data) as fp:
            dm_dict.update(json.load(fp))
        return dm_dict

    else:
        err = f"'dmn_data' is not correct. Must be file or None.\nCurrent value is {dmn_data=}."
        log.error(err)
        raise SystemExit
    

def pta_data_importer(pta_data: str | dict) -> tuple[list[Pulsar], dict | None, array_like | None]:
    """Import PTA pulsars objects, white noise parameters, and empirical distributions.

    Parameters
    ----------
    pta_data : str | dict
        * If string, must be one of ["NG15", "NG12", "IPTA2"].
        * If dict, must have keys ["psrs_data", "noise_data", "emp_dist"]

    Returns
    -------
    psrs : list[Pulsar]
        List of Pulsar objects
    params : dict | None
        Dictionary containing noise data
    emp_dist : array_like | None
        The empirical distribution to use for sampling

    Raises
    ------
    SystemExit
        If `pta_data` is not str or dict.

    """
    if pta_data == "NG15":
        # This returns a path in the astropy cache that points to these files, otherwise
        # it downloads them there and returns the path

        ng15_dic = {
            "psrs_data": download_file(
                "https://zenodo.org/record/8102748/files/ng15_psrs_v1p1.pkl.gz?download=1",
                show_progress=True,
                cache=True,
                pkgname="ptarcade",
            ),
            "noise_data": download_file(
                "https://zenodo.org/record/8102748/files/ng15_wn_v1p1.json?download=1",
                show_progress=True,
                cache=True,
                pkgname="ptarcade",
            ),
            "emp_dist": download_file(
                "https://zenodo.org/record/8102748/files/ng15_emp_v1p1.pkl?download=1",
                show_progress=True,
                cache=True,
                pkgname="ptarcade",
            ),
            "chrom_dict": None,
            "red_dict": None,
            "dm_dict": None,
        }

        psrs = get_pulsars(ng15_dic["psrs_data"])
        params = get_wn(ng15_dic["noise_data"])
        emp_dist = ng15_dic["emp_dist"]
        chrom_dict = get_cn(ng15_dic["chrom_dict"])
        red_dict = get_rn(ng15_dic["red_dict"])
        dm_dict = get_dmn(ng15_dic["dm_dict"])

    elif pta_data == "NG12":
        ng12_dic = {
            "psrs_data": download_file(
                "https://zenodo.org/record/8092873/files/ng12_psrs_v4.pkl.gz?download=1",
                cache=True,
                pkgname="ptarcade",
            ),
            "noise_data": download_file(
                "https://zenodo.org/record/8092873/files/ng12_wn_v4.json?download=1",
                cache=True,
                pkgname="ptarcade",
            ),
            "emp_dist": None,
            "chrom_dict": None,
            "red_dict": None,
            "dm_dict": None,
        }

        psrs = get_pulsars(ng12_dic["psrs_data"])
        params = get_wn(ng12_dic["noise_data"])
        emp_dist = ng12_dic["emp_dist"]
        chrom_dict = get_cn(ng12_dic["chrom_dict"])
        red_dict = get_rn(ng12_dic["red_dict"])
        dm_dict = get_dmn(ng12_dic["dm_dict"])

    elif pta_data == "IPTA2":
        ipta2_dic = {
            "psrs_data": download_file(
                "https://zenodo.org/record/8092873/files/ipta2_psrs_de438.pkl.gz?download=1",
                cache=True,
                pkgname="ptarcade",
            ),
            "noise_data": download_file(
                "https://zenodo.org/record/8092873/files/ipta2_wn_de438.json?download=1",
                cache=True,
                pkgname="ptarcade",
            ),
            "emp_dist": None,
            "chrom_dict": None,
            "red_dict": None,
            "dm_dict": None,
        }

        psrs = get_pulsars(ipta2_dic["psrs_data"])
        params = get_wn(ipta2_dic["noise_data"])
        emp_dist = ipta2_dic["emp_dist"]
        chrom_dict = get_cn(ipta2_dic["chrom_dict"])
        red_dict = get_rn(ipta2_dic["red_dict"])
        dm_dict = get_dmn(ipta2_dic["dm_dict"])
    
    elif pta_data == "EPTA2_full" or pta_data == "EPTA2_new":
        # This returns a path in the astropy cache that points to these files, otherwise
        # it downloads them there and returns the path

        version = pta_data.split('_')[-1]

        epta2_dic = {
            "psrs_data": download_file(
                f"https://zenodo.org/record/8102748/files/epta2_{version}_psrs.pkl.gz?download=1",
                show_progress=True,
                cache=True,
                pkgname="ptarcade",
            ),
            "noise_data": download_file(
                f"https://zenodo.org/record/8102748/files/epta2_{version}_wn.json?download=1",
                show_progress=True,
                cache=True,
                pkgname="ptarcade",
            ),
            "emp_dist": download_file(
                f"https://zenodo.org/record/8102748/files/epta2_{version}_emp.pkl?download=1",
                show_progress=True,
                cache=True,
                pkgname="ptarcade",
            ),
            "chrom_dict": download_file(
                f"https://zenodo.org/record/8102748/files/epta2_{version}_cn.pkl?download=1",
                show_progress=True,
                cache=True,
                pkgname="ptarcade",
            ),
            "red_dict": download_file(
                f"https://zenodo.org/record/8102748/files/epta2_{version}_rn.pkl?download=1",
                show_progress=True,
                cache=True,
                pkgname="ptarcade",
            ),
            "dm_dict": download_file(
                f"https://zenodo.org/record/8102748/files/epta2_{version}_dm.pkl?download=1",
                show_progress=True,
                cache=True,
                pkgname="ptarcade",
            ),
        }

        psrs = get_pulsars(epta2_dic["psrs_data"])
        params = get_wn(epta2_dic["noise_data"])
        emp_dist = epta2_dic["emp_dist"]
        chrom_dict = get_cn(epta2_dic["chrom_dict"])
        red_dict = get_rn(epta2_dic["red_dict"])
        dm_dict = get_dmn(epta2_dic["dm_dict"])

    elif isinstance(pta_data, dict):
        psrs = get_pulsars(pta_data["psrs_data"])
        params = get_wn(pta_data["noise_data"])
        emp_dist = pta_data["emp_dist"]
        chrom_dict = get_cn(pta_data["chrom_dict"])
        red_dict = get_rn(pta_data["red_dict"])
        dm_dict = get_dmn(pta_data["dm_dict"])

    else:
        err = f"'pta_data' is not correct\n.Current value is {pta_data=}."
        log.error(err)
        raise SystemExit

    return psrs, params, emp_dist, chrom_dict, red_dict, dm_dict
