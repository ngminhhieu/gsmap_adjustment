import numpy as np
from pandas import read_csv
from netCDF4 import Dataset
import os
import glob
"""GET NAME, LON, LAT, HEIGHT OF GAUGE DATA"""


def get_lon_lat_gauge_data():

    preprocessed_data_dir = './data/conv2d_gsmap/preprocessed_gauge_data/'
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

    if not os.path.exists('data/ann'):
        os.makedirs('data/ann')
    np.savetxt('data/ann/gauge.csv', precipitation, delimiter=',')
    return lat_arr, lon_arr, precipitation

def get_gsmap_data():
    map_lat = []
    map_lon = []
    map_precip = np.zeros(shape=(1766, 72))

    gauge_lat, gauge_lon, gauge_precip = get_lon_lat_gauge_data()
    for i in range(0, len(gauge_lat)):
        print(i)
        os.system(
            'cdo -outputtab,value -remapnn,lon={}_lat={} data/conv2d_gsmap/gsmap_2011_2018.nc > data/conv2d_gsmap/remapnn.csv'
            .format(gauge_lon[i], gauge_lat[i]))

        gsmap_precipitation = read_csv('data/conv2d_gsmap/remapnn.csv')
        gsmap_precipitation = gsmap_precipitation.to_numpy()
        # *24 because gsmap is measured by average mm/hour
        gsmap_precipitation = gsmap_precipitation * 24
        map_precip[:, i] = gsmap_precipitation[:, 0]

    map_precip = np.round(map_precip, 1)  # round 1 because of gauge_data
    np.savetxt('data/ann/gsmap.csv', map_precip, delimiter=',')

if __name__ == '__main__':
    get_gsmap_data()