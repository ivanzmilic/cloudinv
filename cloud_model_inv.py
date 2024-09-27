import numpy as np
import matplotlib.pyplot as plt 
from astropy.io import fits
from scipy.special import wofz
from scipy.optimize import minimize

import cloud_model as cm
from tqdm import tqdm

from threadpoolctl import threadpool_limits

import os
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
threadpool_limits(1)
import pickle
from enum import IntEnum
from mpi4py import MPI
import sys
threadpool_limits(1)

class tags(IntEnum):
    """ Class to define the state of a worker.
    It inherits from the IntEnum class """ # Makes sense to me, but not sure what is the IntEnum class 
    READY = 0
    DONE = 1
    EXIT = 2
    START = 3

def slice_tasks(datain, task_start, grain_size):
    
    task_end = min(task_start + grain_size, datain.shape[0])

    sl = slice(task_start, task_end) # this is a slice object, allowing us to access the specific thingy
    
    data = {}
    data['taskGrainSize'] = task_end - task_start

    data['spectra'] = datain[sl,:]

    return data





# simple inversion here. to be replaced with the mpi thing:

# load the data:

data = fits.open("/home/milic/data/scratch/mihi_data.fits")[0].data
print ("info::shape of the observed data cube is: ", data.shape)

plt.clf()
plt.figure(figsize=[5,4])
plt.imshow(np.sum(data[:,:,0,200:205], axis =2).T, origin='lower', cmap='grey')
plt.savefig("test_field.png", bbox_inches='tight')

lmin = 45
lmax = 525
ll = fits.open("/home/milic/data/scratch/mihi_data.fits")[1].data[lmin:lmax]

i = 44
j = 54
ll0 = 6552.8

spectrum_to_fit = data[i,j,0,lmin:lmax]
result = cm.invert_simple_py(ll, spectrum_to_fit, ll0, 1E-4)

print (result.x)

