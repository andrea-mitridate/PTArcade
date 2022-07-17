import os
import pickle
import glob, json
from enterprise.pulsar import Pulsar

cwd = os.getcwd()
pta_dat_dir = cwd + '/inputs/pta/pta_data/'

def get_pulsars(pta_data, ephemeris, filter=None):

    if os.path.isfile(pta_dat_dir + pta_data):
        with open(pta_dat_dir + pta_data, 'rb') as handle:
            psrs = pickle.load(handle)
        
        return psrs

    elif os.path.isdir(pta_dat_dir + pta_dat_dir):
        parfiles = sorted(glob.glob(pta_dat_dir + pta_dat_dir + '/par/*.par'))
        timfiles = sorted(glob.glob(pta_dat_dir + pta_dat_dir + '/tim/*.tim'))

        # filter
        if filter is not None:
            parfiles = [x for x in parfiles if x.split('/')[-1].split('.')[0] in filter]
            timfiles = [x for x in timfiles if x.split('/')[-1].split('.')[0] in filter]

        # Make sure you use the tempo2 parfile for J1713+0747!!
        # ...filtering out the tempo parfile... 
        parfiles = [x for x in parfiles if 'J1713+0747_NANOGrav_12yv3.gls.par' not in x]

        # load the pulsars into enterprise 
        psrs = []
        for p, t in zip(parfiles, timfiles):
            psr = Pulsar(p, t, ephem=ephemeris)
            psrs.append(psr)
    

        return psrs


def get_wn(wn_data):
    
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
