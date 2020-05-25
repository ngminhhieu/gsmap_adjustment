from netCDF4 import Dataset
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
import math
import glob
from pandas import read_csv

def get_lon_lat_gauge_data():
    
    preprocessed_data_dir = './data/conv2d_gsmap/txt_data/'
    start_index = len(preprocessed_data_dir)
    type_file = '.csv'
    end_index = -len(type_file)
    data_paths_preprocessed_data = glob.glob(preprocessed_data_dir + '*' +
                                             type_file)

    lon_arr = np.empty(shape=(len(data_paths_preprocessed_data)))
    lat_arr = np.empty(shape=(len(data_paths_preprocessed_data)))
    precipitation = np.zeros(shape=(1766,72))
    for index, file in enumerate(data_paths_preprocessed_data):
        file_name = file[start_index:end_index]
        file_name_list = file_name.split('_')
        lon = float(file_name_list[1])
        lat = float(file_name_list[2])
        lon_arr[index] = lon
        lat_arr[index] = lat
        precip = read_csv(file, usecols=['precipitation'])
        precip = precip.to_numpy()
        precipitation[:, index] = precip[-1766:, 0]
    lon_arr = np.round(lon_arr, 3)
    lat_arr = np.round(lat_arr, 3)

    return lat_arr, lon_arr, precipitation

nc = Dataset('data/conv2d_gsmap/gsmap_2011_2018.nc', 'r')
time = np.array(nc['time'][:])
map_lon = np.array(nc['lon'][:])
map_lat = np.array(nc['lat'][:])
map_precip = np.array(nc['precip'][:])
map_precip = map_precip*24

gauge_lat, gauge_lon, gauge_precip = get_lon_lat_gauge_data()

print(map_lon.shape)
print(map_lat.shape)
print(map_precip.shape)
print(gauge_lon.shape)
print(gauge_lat.shape)
print(gauge_precip.shape)


print(np.count_nonzero(gauge_precip < 0))


np.savez('data/conv2d_gsmap/npz/raw_map_gauge_data.npz',
             time=time,
             map_lat=map_lat,
             map_lon=map_lon,
             map_precip=map_precip,
             gauge_lat=gauge_lat,
             gauge_lon=gauge_lon,
             gauge_precip=gauge_precip)