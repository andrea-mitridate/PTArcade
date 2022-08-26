import os
import pickle
import glob, json
from enterprise.pulsar import Pulsar

cwd = os.getcwd()
pta_dat_dir = os.path.join(cwd, 'inputs/pta_data/')

ng15_dic = {
    'psrs_data': 'ng15_psrs_v1p1.pkl',
    'noise_data': 'ng15_wn_v1p1.json',
    'emp_dist': 'ng15_emp_v1p1.pkl'}

ng12_dic = {
    'psrs_data': 'ng12_psrs_v4.pkl',
    'noise_data': 'ng12_psrs_v4.json',
    'emp_dist': None}


def get_ephem_conv(par_file):
    """
    get the ephemeris convention used in par files.
    """
    f = open(par_file, 'r')
    lines = f.readlines()

    for line in lines:
        line = line.split()
        if line[0] == 'EPHEM':
            ephem = line[-1]
    
    f.close()

    return ephem


def get_pulsars(pta_data, filter=None):

    pta_data = os.path.join(pta_dat_dir, pta_data)

    if os.path.isfile(pta_data):
        with open(pta_data, 'rb') as handle:
            psrs = pickle.load(handle)

        return psrs

    elif os.path.isdir(pta_data):
        parfiles = sorted(glob.glob(os.path.join(pta_data, '*.par')))
        timfiles = sorted(glob.glob(os.path.join(pta_data, '*.tim')))

        # filter
        if filter is not None:
            parfiles = [x for x in parfiles if x.split('/')[-1].split('.')[0] in filter]
            timfiles = [x for x in timfiles if x.split('/')[-1].split('.')[0] in filter]

        # load the pulsars into enterprise 
        psrs = []
        for p, t in zip(parfiles, timfiles):
            ephemeris = get_ephem_conv(p)
            psr = Pulsar(p, t, ephem=ephemeris)
            psrs.append(psr)
    
        return psrs


def get_wn(wn_data):
    if wn_data is None:
        return None
    
    params = {}

    if os.path.isfile(pta_dat_dir + wn_data):
        with open(pta_dat_dir + wn_data, 'r') as fp:
            params.update(json.load(fp))

        return params

    elif os.path.isdir(pta_dat_dir + wn_data):
        for filename in os.listdir(pta_dat_dir + wn_data):
            with open(os.path.join(pta_dat_dir + wn_data, filename), 'r') as f:
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
        emp_dist = os.path.join(pta_dat_dir, ng15_dic['emp_dist'])
    elif pta_data == 'NG12':
        psrs = get_pulsars(ng12_dic['psrs_data'])
        params = get_wn(ng12_dic['noise_data'])
        emp_dist = os.path.join(pta_dat_dir, ng12_dic['emp_dist'])
    else:
        psrs = get_pulsars(pta_data['psrs_data'])
        params = get_wn(pta_data['noise_data'])
        emp_dist = os.path.join(pta_dat_dir, pta_data['emp_dist'])

    return psrs, params, emp_dist
