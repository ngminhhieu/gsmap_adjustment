import numpy as np
from model import common_util
from sklearn.preprocessing import MinMaxScaler
from keras import backend as K

def create_data_prediction(**kwargs):

    data_npz = kwargs['data'].get('dataset')
    seq_len = kwargs['model'].get('seq_len')
    horizon = kwargs['model'].get('horizon')

    time = np.load(data_npz)['time']
    # horizon is in seq_len. the last
    T = len(time)

    map_lon = np.load(data_npz)['map_lon']
    map_lat = np.load(data_npz)['map_lat']
    map_precip = np.load(data_npz)['map_precip']
    map_cloud_cover = np.load(data_npz)['map_cloud_cover']
    map_sea_level = np.load(data_npz)['map_sea_level']
    map_surface_temp = np.load(data_npz)['map_surface_temp']
    map_wind_u_mean = np.load(data_npz)['map_wind_u_mean']
    map_wind_v_mean = np.load(data_npz)['map_wind_v_mean']
    gauge_precip = np.load(data_npz)['gauge_precip']

    # input is gsmap
    input_model = np.zeros(shape=(T, 160, 120, 5))
    # output is gauge
    output_model = np.zeros(shape=(T, 160, 120, 1))

    for i in range(len(map_lat)):
        lat = map_lat[i]
        lon = map_lon[i]
        temp_lat = int(round((23.95 - lat) / 0.1))
        temp_lon = int(round((lon - 100.05) / 0.1))
        input_model[:, temp_lat, temp_lon, 0] = map_precip[:, i]
        input_model[:, temp_lat, temp_lon, 1] = map_wind_u_mean[:, i]
        input_model[:, temp_lat, temp_lon, 2] = map_wind_v_mean[:, i]
        input_model[:, temp_lat, temp_lon, 3] = map_surface_temp[:, i]
        # input_model[:, temp_lat, temp_lon, 4] = map_cloud_cover[:, i]
        input_model[:, temp_lat, temp_lon, 5] = map_sea_level[:, i]
        output_model[:, temp_lat, temp_lon, 0] = gauge_precip[:, i]
        
    return input_model, output_model


def load_dataset(**kwargs):
    # get preprocessed input and target
    input_conv2d_gsmap, target_conv2d_gsmap = create_data_prediction(**kwargs)

    # get test_size, valid_size from config
    test_size = kwargs['data'].get('test_size')
    valid_size = kwargs['data'].get('valid_size')

    # split data to train_set, valid_set, test_size
    input_train, input_valid, input_test = common_util.prepare_train_valid_test(
        input_conv2d_gsmap, test_size=test_size, valid_size=valid_size)
    target_train, target_valid, target_test = common_util.prepare_train_valid_test(
        target_conv2d_gsmap, test_size=test_size, valid_size=valid_size)
    data = {}
    for cat in ["train", "valid", "test"]:
        x, y = locals()["input_" + cat], locals()["target_" + cat]
        data["input_" + cat] = x
        data["target_" + cat] = y

    return data