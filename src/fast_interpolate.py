# Read data from an HDF5 file and use it for interpolation of the spectrum.
# The file must have the following data sets:
# 'parameter_names': array of strings giving the names of the parameters in order.
# The last parameter must be the gravitational wave frequency 'f'.
# Each parameter name: array giving start and step values.
# 'spectrum': array indexed by parameters in order, giving omega_GW for these parameters.

import numpy as np
import h5py

# Load HDF5 file.  Returns info: list of (name, start, step)
# and multidimensional array of data points
def load_data(file):
    with h5py.File(file) as h5:
        info = [(par, h5[par][0], h5[par][1]) for par in h5['parameter_names'].asstr()]
        spectrum = np.array(h5['spectrum'])
        return (info, spectrum)

# Do interpolation
# Called with info: list of (start, step, value)
# and multidimensional array of data points
# Multiple values (in an np.array) are OK for the last value, but not earlier ones
# or else we will get confusion about array indexes.
def interp(info, data):
    if len(info) == 0:       # Nothing to do: just return element
        return data
    x0, dx, x = info[0]
    (fract, index) = np.modf((x - x0) / dx)
    index = index.astype(int)
    # Call ourselves to interpolate over remaining variables if any
    # then combine results linearly
    return (interp(info[1:], data[index]) * (1-fract)
            + interp(info[1:], data[index+1]) * fract)


# Convert an old-style file to our HDF5 format.
# The old-style has a header giving the parameter names including 'spectrum', then
# on each line the parameter values and the resulting omega_GW.
# The last parameter must vary fastest, and for each earlier parameter, the later parameters
# must go through precisely the same values.  The values must be evenly spaced
# and in ascending order.
# There is no error checking in this code!
def reformat(infile, outfile):
    par_names = np.loadtxt(infile, max_rows=1, dtype='str') # Parameter names
    data = np.loadtxt(infile, skiprows=1)
    spec_col = np.where(par_names=='spectrum')[0] # Index of spectrum
    spectrum = data.T[spec_col]              # List of data
    data = np.delete(data, spec_col, axis=1)      # Remove data
    par_names = np.delete(par_names, spec_col)    # and 'spectrum'
    shape = np.zeros(len(par_names))              # This will be shape of data array
    with h5py.File(outfile,'w') as out:
        out.create_dataset("parameter_names", data=par_names.tolist())
        for idx, par in enumerate(par_names):
            if par == 'f' and idx != len(par_names)-1:
                raise Exception("f field should have been last") # Otherwise interp can't do multiple probes
            values = np.sort(np.unique(data.T[idx]))
            dataset = out.create_dataset(par, data=[values[0], values[1]-values[0]]) # Start and step
            shape[idx] = len(values)
        out.create_dataset("spectrum", data=spectrum, shape=shape) # Reshape and write
