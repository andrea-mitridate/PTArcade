"""Import PTA data."""
from __future__ import annotations

import glob
import json
import os
import pickle

from astropy.utils.data import download_file
from enterprise.pulsar import Pulsar
from numpy._typing import _ArrayLikeFloat_co as array_like


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
    f = open(par_file, "r")
    lines = f.readlines()

    for line in lines:
        line = line.split()
        if line[0] == "EPHEM":
            ephem = line[-1]

    f.close()

    return ephem


def get_pulsars(pta_data: str, filters: list[str] = None) -> list[Pulsar]:
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

    """
    if os.path.isfile(pta_data):
        with open(pta_data, "rb") as handle:
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
    if wn_data is None:
        return None

    params = {}

    if os.path.isfile(wn_data):
        with open(wn_data, 'r') as fp:
            params.update(json.load(fp))

        return params

    elif os.path.isdir(wn_data):
        for filename in os.listdir(wn_data):
            with open(os.path.join(wn_data, filename), 'r') as f:
                data = json.load(f)
                for key, value in data.items():
                    params.update({key: value})

        return params


def pta_data_importer(pta_data):
    """
    Import pta pulsars objects, white noise parameters, and empirical distributions.
    """
    if pta_data == 'NG15':
        psrs = get_pulsars(ng15_dic['psrs_data'])
        params = get_wn(ng15_dic['noise_data'])
        emp_dist = ng15_dic['emp_dist']
    elif pta_data == 'NG12':
        psrs = get_pulsars(ng12_dic['psrs_data'])
        params = get_wn(ng12_dic['noise_data'])
        emp_dist = ng12_dic['emp_dist']
    elif pta_data == 'IPTA2':
        psrs = get_pulsars(ipta2_dic['psrs_data'])
        params = get_wn(ipta2_dic['noise_data'])
        emp_dist = ipta2_dic['emp_dist']
    else:
        psrs = get_pulsars(pta_data['psrs_data'])
        params = get_wn(pta_data['noise_data'])
        emp_dist = pta_data['emp_dist']

    if emp_dist is not None:
        emp_dist = os.path.join(pta_dat_dir, emp_dist)

    return psrs, params, emp_dist
