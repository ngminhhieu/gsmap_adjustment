import numpy as np
from pandas import read_csv
from netCDF4 import Dataset
import os
import glob

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
    gauge_lat_arr = []
    gauge_lon_arr = []
    gauge_precip_arr = np.empty(shape=(1766, 72))
    gsmap_precip_arr = np.empty(shape=(1766, 72))
    # get precipitation
    lat_arr, lon_arr, precipitation = get_lon_lat_gauge_data()
    for i in range(0, len(lat_arr)):
        os.system(
            'cdo -outputtab,value -remapnn,lon={}_lat={} data/conv2d_gsmap/gsmap_2011_2018.nc > data/conv2d_gsmap/remapnn.csv'
            .format(lon_arr[i], lat_arr[i]))
        lat, lon = find_lat_lon_remapnn()
        gauge_lat_arr.append(lat)
        gauge_lon_arr.append(lon)
        precip = precipitation[i]
        gauge_precip_arr[:, i] = precip[:, 0]

        gsmap_precipitation = read_csv('data/conv2d_gsmap/remapnn.csv')
        gsmap_precipitation = gsmap_precipitation.to_numpy()
        gsmap_precip_arr[:, i] = gsmap_precipitation[:, 0]

    cal_error(gauge_precip_arr[-353:,:], gsmap_precip_arr[-353:,:])

    return gauge_lat_arr, gauge_lon_arr, gauge_precip_arr

def save_to_npz():
    gauge_lat, gauge_lon, gauge_precip = set_gauge_data_to_gsmap()

    np.savez('data/conv2d_gsmap/npz/gauge_data.npz',
             gauge_lon=gauge_lon,
             gauge_lat=gauge_lat,
             gauge_precip=gauge_precip)

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

save_to_npz()