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

# series = read_csv('./data/ann/gsmap.csv', names=range(0, 72))
# trend = np.empty(shape=(len(series), len(series.columns)))
# seasonal = np.empty(shape=(len(series), len(series.columns)))
# print(trend.shape)
# series.index = pd.DatetimeIndex(freq='w', start=0, periods=len(series))
# for i in range(len(series.columns)):
#     result = seasonal_decompose(series[i])
#     trend[:, i] = result.trend.fillna(0)
#     seasonal[:, i] = result.seasonal

# np.savetxt('./data/ann/precip_seasonal.csv', trend, delimiter=',')
# np.savetxt('./data/ann/precip_trend.csv', trend, delimiter=',')

# data = np.empty(shape=(200, 3))
# for i in range(len(data)):
#     data[i, 0] = i
#     data[i, 1] = i%4

# for ran in range(50):
#     ran = random.randint(0, 10)
#     data[ran, 0] = ran

# data[0:20, 2] = data[0:20, 1]
# data[20:100, 2] = data[20:100, 1] + 2
# data[100:200, 2] = data[100:200, 1] -3
# np.savetxt('./data/ann/test.csv', data, delimiter=',')

# for i in range(3):
#     plt.plot(data[:, i])
#     plt.show()

a = np.array([[1,2,4], [5,6,10], [4,1,2]])
print(np.reshape(np.ravel(a, order='F'), (-1,1)))