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


def cal_error_gauge_gsmap():
    dataset = 'data/conv2d_gsmap/npz/map_gauge_72_stations.npz'
    map_precip = np.load(dataset)['map_precip']
    gauge_precip = np.load(dataset)['gauge_precip']
    cal_error(gauge_precip[-354:, :], map_precip[-354:, :])


def cal_error(test_arr, prediction_arr):
    from sklearn.metrics import mean_squared_error, mean_absolute_error
    with np.errstate(divide='ignore', invalid='ignore'):
        # cal mse
        error_mae = mean_absolute_error(test_arr, prediction_arr)

        # cal rmse
        error_mse = mean_squared_error(test_arr, prediction_arr)
        error_rmse = np.sqrt(error_mse)

        # cal mape
        y_true, y_pred = np.array(test_arr), np.array(prediction_arr)
        error_mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
        error_list = [error_mae, error_rmse, error_mape]
        print("MAE: %.4f" % (error_mae))
        print("RMSE: %.4f" % (error_rmse))
        print("MAPE: %.4f" % (error_mape))
        return error_list


# save_to_npz()
# cal_error_gauge_gsmap()

def get_raw_precip():
    original_nc = Dataset('data/conv2d_gsmap/gsmap_2011_2018.nc', 'r')

    # check dataset
    gsmap_precip = np.array(original_nc['precip'][:])
    gsmap_precip = np.round(gsmap_precip, 1)

    npz_url = 'data/conv2d_gsmap/map_gauge_72_stations.npz'
    time = np.load(npz_url)['time']
    map_lon = np.load(npz_url)['map_lon']
    map_lat = np.load(npz_url)['map_lat']
    map_precip = np.load(npz_url)['map_precip']

    gauge_lon = np.load(npz_url)['gauge_lon']
    gauge_lat = np.load(npz_url)['gauge_lat']
    gauge_precip = np.load(npz_url)['gauge_precip']

    abc = gsmap_precip.shape[1]*gsmap_precip.shape[2]
    rain_gsmap = np.zeros(shape=(1766, abc))
    for lat in range(gsmap_precip.shape[1]):
        for lon in range(gsmap_precip.shape[2]):
            rain_gsmap[:, lat*120+lon] = gsmap_precip[:, lat, lon]
    
    np.savez('data/conv2d_gsmap/map_gauge_72_stations.npz',
             time=time,
             map_lat=map_lat,
             map_lon=map_lon,
             map_precip=map_precip,
             gauge_lat=gauge_lat,
             gauge_lon=gauge_lon,
             gauge_precip=gauge_precip,
             raw_precip_gsmap=rain_gsmap)

get_raw_precip()