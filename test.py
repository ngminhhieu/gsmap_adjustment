import numpy as np
from netCDF4 import Dataset

original_nc = Dataset('data/conv2d_gsmap/gsmap_2011_2018.nc', 'r')

# check dataset
print(np.array(original_nc['lat'][:]))
print(np.array(original_nc['lon'][:]))