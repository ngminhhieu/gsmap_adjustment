import numpy as np
from pandas import read_csv
from netCDF4 import Dataset
import os
import glob
"""GET NAME, LON, LAT, HEIGHT OF GAUGE DATA"""


def get_lon_lat_gauge_data():

    preprocessed_data_dir = './data/preprocessed_txt_data/'
    start_index = len(preprocessed_data_dir)
    type_file = '.csv'
    end_index = -len(type_file)
    data_paths_preprocessed_data = glob.glob(preprocessed_data_dir + '*' +
                                             type_file)

    lon_arr = np.empty(shape=(len(data_paths_preprocessed_data)))
    lat_arr = np.empty(shape=(len(data_paths_preprocessed_data)))
    for index, file in enumerate(data_paths_preprocessed_data):
        file_name = file[start_index:end_index]
        file_name_list = file_name.split('_')
        lon = float(file_name_list[1])
        lat = float(file_name_list[2])
        lon_arr[index] = lon
        lat_arr[index] = lat

    lon_arr = np.round(lon_arr, 3)
    lat_arr = np.round(lat_arr, 3)

    return lat_arr, lon_arr


def find_lat_lon_remapnn():
    # find nearest lon_lat
    precip_list = read_csv('./data/remapnn.csv')
    precip_list = precip_list.to_numpy()

    original_nc = Dataset('data/gsmap_2011_2018.nc', 'r')

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
    input_lat_arr = []
    input_lon_arr = []

    lat_arr, lon_arr = get_lon_lat_gauge_data()
    for i in range(0, len(lat_arr)):
        os.system(
            'cdo -outputtab,value -remapnn,lon={}_lat={} data/gsmap_2011_2018.nc > data/remapnn.csv'
            .format(lon_arr[i], lat_arr[i]))
        lat, lon = find_lat_lon_remapnn()
        input_lat_arr.append(lat)
        input_lon_arr.append(lon)

    input_lat_arr.sort()
    input_lon_arr.sort()

    # get precipitation
    input_precip_arr = np.empty(shape=(1766,
                                       len(input_lat_arr), len(input_lon_arr)))
    for i in range(0, len(input_lat_arr)):
        os.system(
            'cdo -outputtab,value -remapnn,lon={}_lat={} data/conv2d_gsmap/gsmap_2011_2018.nc > data/precip.csv'
            .format(input_lon_arr[i], input_lat_arr[i]))
        precipitation = read_csv('data/precip.csv')
        precipitation = precipitation.to_numpy()
        input_precip_arr[:, i, i] = precipitation[:,0]

    return input_lat_arr, input_lon_arr, input_precip_arr


def save_to_npz():
    input_lat, input_lon, input_precip = set_gauge_data_to_gsmap()
    output_nc = Dataset('data/gsmap_2011_2018.nc', 'r')

    time = np.array(output_nc['time'][:])
    output_lon = np.array(output_nc['lon'][:])
    output_lat = np.array(output_nc['lat'][:])
    output_precip = np.array(output_nc['precip'][:])

    np.savez('data/npz/conv2d_gsmap.npz',
             time=time,
             input_lon=input_lon,
             input_lat=input_lat,
             input_precip=input_precip,
             output_lon=output_lon,
             output_lat=output_lat,
             output_precip=output_precip)

def test():
    input_precip = np.load('data/npz/conv2d_gsmap.npz')['input_precip']
    print(input_precip[4,70,70])

# test()
save_to_npz()