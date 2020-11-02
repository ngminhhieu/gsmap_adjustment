from netCDF4 import Dataset
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
import math
import glob
from pandas import read_csv

def get_lon_lat_gauge_data():
    
    preprocessed_data_dir = './data/conv2d_gsmap/preprocessed_txt_data/'
    start_index = len(preprocessed_data_dir)
    type_file = '.csv'
    end_index = -len(type_file)
    data_paths_preprocessed_data = glob.glob(preprocessed_data_dir + '*' +
                                             type_file)

    lon_arr = np.empty(shape=(len(data_paths_preprocessed_data)))
    lat_arr = np.empty(shape=(len(data_paths_preprocessed_data)))
    precipitation = []
    for index, file in enumerate(data_paths_preprocessed_data):
        file_name = file[start_index:end_index]
        file_name_list = file_name.split('_')
        lon = float(file_name_list[1])
        lat = float(file_name_list[2])
        lon_arr[index] = lon
        lat_arr[index] = lat
        precip = read_csv(file, usecols=['precipitation'])
        precip = precip.to_numpy()
        precipitation.append(precip[-1766:])
    lon_arr = np.round(lon_arr, 3)
    lat_arr = np.round(lat_arr, 3)

    return lat_arr, lon_arr, precipitation

nc = Dataset('data/conv2d_gsmap/gsmap_2011_2018.nc', 'r')
time = np.array(nc['time'][:])
lon = np.array(nc['lon'][:])
lat = np.array(nc['lat'][:])
precip = np.array(nc['precip'][:])

lon_plot = []
lat_plot = []

nearest_dataset = 'data/conv2d_gsmap/npz/gauge_data.npz'
nearest_lon = np.load(nearest_dataset)['gauge_lon']
nearest_lat = np.load(nearest_dataset)['gauge_lat']
nearest_precipitation = np.load(nearest_dataset)['nearest_precip']

for i in range(len(lon)):
    temp_lon = np.repeat(lon[i], len(lat))
    lon_plot.append(temp_lon)
    for i in range(len(lat)):
        lat_plot.append(lat[i])


plt.scatter(lon_plot,lat_plot, s=0.5, c='green')
plt.scatter(nearest_lon, nearest_lat, s=1, c='red')
# plt.show()

gauge_lat, gauge_lon, gauge_precip = get_lon_lat_gauge_data()

number_of_gauges = len(gauge_lat)

def distance(lat_gauge, lon_gauge, lat_map, lon_map):
    return math.sqrt((lat_gauge-lat_map)**2 + (lon_gauge-lon_map)**2)


gauge_precip = np.array(gauge_precip)
gauge_precip = np.reshape(gauge_precip, (1766,72))

nearest_precipitation = nearest_precipitation[-361:, :]
gauge_precip =  gauge_precip[-361:, :]
adjusted_precip = np.zeros(shape=(354, 72))
for time in range(len(nearest_precipitation)-7):
    total_distance = 0
    bias = 0
    for i in range(number_of_gauges):
        R_i_t = np.mean(nearest_precipitation[time:time+7, i])
        G_i_t = np.mean(gauge_precip[time:time+7, i])
        if R_i_t == 0.0 and G_i_t == 0.0:
            R_i_t = 1
            G_i_t = 1
        elif R_i_t == 0.0:
            R_i_t = 0.1
        elif G_i_t == 0.0:
            G_i_t = 0.1
        bias_gauge_map = G_i_t/R_i_t
        
        d_i = distance(gauge_lat[i], gauge_lon[i], nearest_lat[i], nearest_lon[i])

        total_distance = total_distance + d_i
        bias_gm = d_i * 10 * math.log10(bias_gauge_map)
        bias = bias + bias_gm
    
    F = math.pow(10, (1/total_distance) * bias/10)
    adjusted_precip[time, :] = nearest_precipitation[time+7, :] * F


from sklearn.metrics import mean_squared_error, mean_absolute_error
def mae(test_arr, prediction_arr):
    with np.errstate(divide='ignore', invalid='ignore'):
        error_mae = mean_absolute_error(test_arr, prediction_arr)
        print("MAE: %.4f" % (error_mae))
        return error_mae

mae(gauge_precip[-354:,], adjusted_precip)