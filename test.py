import numpy as np

# data_npz = './data/conv2d_gsmap/all_data.npz'

# map_lon = np.load(data_npz)['map_lon']
# map_lat = np.load(data_npz)['map_lat']
# map_precip = np.load(data_npz)['map_precip']
# map_cloud_cover = np.load(data_npz)['map_cloud_cover']
# map_sea_level = np.load(data_npz)['map_sea_level']
# map_surface_temp = np.load(data_npz)['map_surface_temp']
# map_wind_u_mean = np.load(data_npz)['map_wind_u_mean']
# map_wind_v_mean = np.load(data_npz)['map_wind_v_mean']
# gauge_precip = np.load(data_npz)['gauge_precip']

# np.savetxt('./data/ann/cloud_cover.csv', map_cloud_cover, delimiter=",")
# np.savetxt('./data/ann/sea_level.csv', map_sea_level, delimiter=",")
# np.savetxt('./data/ann/surface_temp.csv', map_surface_temp, delimiter=",")
# np.savetxt('./data/ann/wind_u_mean.csv', map_wind_u_mean, delimiter=",")
# np.savetxt('./data/ann/wind_v_mean.csv', map_wind_v_mean, delimiter=",")

import pandas as pd
from pandas import read_csv
from matplotlib import pyplot
import numpy as np
from statsmodels.tsa.seasonal import seasonal_decompose
from matplotlib import pyplot as plt
import random
import scipy.stats
import netCDF4 as nc

ds = nc.Dataset('./data/conv2d_gsmap/gsmap_2011_2018.nc')
gsmap_precip = ds['precip'][:]
gsmap_lat = ds['lat'][:]
gsmap_lon = ds['lon'][:]
gauge_data = read_csv('./data/ann/gauge.csv', header=None).to_numpy()
correlation = np.empty(shape=(gauge_data.shape[0], gauge_data.shape[1]))
max_corr = 0
max_lat = 0
max_lon = 0
for gauge_index in range(gauge_data.shape[1]):
    gauge_precip = gauge_data[:, gauge_index]
    for lat_index in range(len(gsmap_lat)):
        for lon_index in range(len(gsmap_lon)):
            map_precip = gsmap_precip[:, lat_index, lon_index]
            r = np.corrcoef(map_precip, gauge_precip)
            r = r[0, 1]
            if abs(max_corr) <= abs(r):
                max_corr = r
                max_lat = lat_index
                max_lon = lon_index
    print(max_corr)
    correlation[:, gauge_index] = gsmap_precip[:, max_lat, max_lon]
    import sys
    sys.exit()

np.savetxt('./data/ann/correlation.csv', correlation , delimiter=",")