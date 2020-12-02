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
series = read_csv('./data/ann/gsmap.csv', names=range(0, 72))
series.index = pd.DatetimeIndex(freq='w', start=0, periods=len(series))
result = seasonal_decompose(series[0])
trend = result.trend
seasonal = result.seasonal
trend = trend.fillna(0)
print(seasonal.isnull().sum(axis = 0))
print(trend.isnull().sum(axis = 0))
trend.to_csv('./data/ann/precip_trend.csv', index=False)
seasonal.to_csv('./data/ann/precip_seasonal.csv', index=False)