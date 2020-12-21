import numpy as np

data_npz = './data/conv2d_gsmap/all_data.npz'

# map_lon = np.load(data_npz)['map_lon']
# map_lat = np.load(data_npz)['map_lat']
map_precip = np.load(data_npz)['map_precip']
# map_cloud_cover = np.load(data_npz)['map_cloud_cover']
# map_sea_level = np.load(data_npz)['map_sea_level']
map_surface_temp = np.load(data_npz)['map_surface_temp']
map_wind_u_mean = np.load(data_npz)['map_wind_u_mean']
map_wind_v_mean = np.load(data_npz)['map_wind_v_mean']
gauge_precip = np.load(data_npz)['gauge_precip']
gauge_lon = np.load(data_npz)['gauge_lon']
gauge_lat = np.load(data_npz)['gauge_lat']

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
import seaborn as sns
from sklearn.cluster import KMeans

# series = read_csv('./data/ann/gsmap.csv', names=range(0, 72))
# trend = np.empty(shape=(len(series), len(series.columns)))
# seasonal = np.empty(shape=(len(series), len(series.columns)))
# print(trend.shape)
# series.index = pd.DatetimeIndex(freq='w', start=0, periods=len(series))
# for i in range(len(series.columns)):
#     result = seasonal_decompose(series[i])
#     trend[:, i] = result.trend.fillna(0)
#     seasonal[:, i] = result.seasonal

# np.savetxt('./data/ann/precip_seasonal.csv', seasonal, delimiter=',')
# np.savetxt('./data/ann/precip_trend.csv', trend, delimiter=',')

gauge_dataset = read_csv('./data/ann/gauge.csv', names=range(0, 72)).to_numpy()
gauge_coordinate = np.empty(shape=(72, 3))

for i in range(len(gauge_lat)):
    gauge_coordinate[i, 0] = gauge_lat[i]
    gauge_coordinate[i, 1] = gauge_lon[i]
    gauge_coordinate[i, 2] = gauge_precip[-1, i]

kmeans = KMeans(n_clusters=3, random_state=0).fit(gauge_coordinate)

count = 0 
for index, value in enumerate(kmeans.labels_):
    if value == 0:
        count += 1

gsmap_dataset_group = np.empty(shape=(len(gauge_dataset), count))
gauge_dataset_group = np.empty(shape=(len(gauge_dataset), count))
wind_u_mean_group = np.empty(shape=(len(gauge_dataset), count))
wind_v_mean_group = np.empty(shape=(len(gauge_dataset), count))
surface_temp_group = np.empty(shape=(len(gauge_dataset), count))
count = 0
labels = kmeans.labels_
for index in range(len(labels)):
    if labels[index] == 0:
        gsmap_dataset_group[:, count] = map_precip[:, index]
        gauge_dataset_group[:, count] = gauge_dataset[:, index]
        wind_u_mean_group[:, count] = map_wind_u_mean[:, index]
        wind_v_mean_group[:, count] = map_wind_v_mean[:, index]
        surface_temp_group[:, count] = map_surface_temp[:, index]
        count += 1

np.savetxt('./data/ann/gsmap_group.csv', gsmap_dataset_group, delimiter=',')
np.savetxt('./data/ann/gauge_group.csv', gauge_dataset_group, delimiter=',')
np.savetxt('./data/ann/wind_u_mean_group.csv', wind_u_mean_group, delimiter=',')
np.savetxt('./data/ann/wind_v_mean_group.csv', wind_v_mean_group, delimiter=',')
np.savetxt('./data/ann/surface_temp_group.csv', surface_temp_group, delimiter=',')

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


# x = np.arange(5.)
# y = np.arange(5.)
# print(scipy.stats.pearsonr(x, y))

# plt.scatter(x, y)
# plt.show()