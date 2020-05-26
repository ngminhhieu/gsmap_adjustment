import numpy as np
from pandas import read_csv
from netCDF4 import Dataset
import os
import glob
"""GET NAME, LON, LAT, HEIGHT OF GAUGE DATA"""


def get_lon_lat_gauge_data():

    preprocessed_data_dir = './data/conv2d_gsmap/preprocessed_txt_data/'
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


def find_lat_lon_remapnn():
    # find nearest lon_lat
    precip_list = read_csv('./data/conv2d_gsmap/remapnn.csv')
    precip_list = precip_list.to_numpy()

    original_nc = Dataset('data/conv2d_gsmap/gsmap_2011_2018.nc', 'r')

    # check dataset
    gsmap_time = np.array(original_nc['time'][:])
    gsmap_lon = np.array(original_nc['lon'][:])
    gsmap_lat = np.array(original_nc['lat'][:])
    gsmap_precip = np.array(original_nc['precip'][:])
    gsmap_precip = np.round(gsmap_precip, 6)

    index_pos = []
    # precip_list - 2 due to the header and 0-based index
    for i in range(0, len(precip_list)):
        if precip_list[i] > 0:
            index_pos = np.where(gsmap_precip[i, :, :] == np.round(
                precip_list[i, 0], 6))  # 0 because of precip_list is (xxx,1)
            if len(index_pos[0]) == 1 and len(index_pos[1]) == 1:
                break

    # get lat and long
    # 0 is lat
    # 1 is lon
    return gsmap_lat[index_pos[0][0]], gsmap_lon[index_pos[1][0]]


def set_gauge_data_to_gsmap():
    map_lat = []
    map_lon = []
    map_precip = np.zeros(shape=(1766, 72))

    gauge_lat, gauge_lon, gauge_precip = get_lon_lat_gauge_data()
    for i in range(0, len(gauge_lat)):
        os.system(
            'cdo -outputtab,value -remapnn,lon={}_lat={} data/conv2d_gsmap/gsmap_2011_2018.nc > data/conv2d_gsmap/remapnn.csv'
            .format(gauge_lon[i], gauge_lat[i]))
        lat_nearest_gauge, lon_nearest_gauge = find_lat_lon_remapnn()
        map_lat.append(lat_nearest_gauge)
        map_lon.append(lon_nearest_gauge)

        gsmap_precipitation = read_csv('data/conv2d_gsmap/remapnn.csv')
        gsmap_precipitation = gsmap_precipitation.to_numpy()
        # *24 because gsmap is measured by average mm/hour
        gsmap_precipitation = gsmap_precipitation * 24
        map_precip[:, i] = gsmap_precipitation[:, 0]
        print(i)

    map_precip = np.round(map_precip, 1)  # round 1 because of gauge_data

    return map_lat, map_lon, map_precip, gauge_lat, gauge_lon, gauge_precip


def save_to_npz():
    map_lat, map_lon, map_precip, gauge_lat, gauge_lon, gauge_precip = set_gauge_data_to_gsmap()
    raw_gsmap = Dataset('data/conv2d_gsmap/gsmap_2011_2018.nc', 'r')
    time = np.array(raw_gsmap['time'][:])

    np.savez('data/conv2d_gsmap/npz/map_gauge_72_stations.npz',
             time=time,
             map_lat=map_lat,
             map_lon=map_lon,
             map_precip=map_precip,
             gauge_lat=gauge_lat,
             gauge_lon=gauge_lon,
             gauge_precip=gauge_precip)

save_to_npz()
