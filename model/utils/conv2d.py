import numpy as np
from model import common_util
from sklearn.preprocessing import MinMaxScaler


def create_data_prediction(**kwargs):

    data_npz = kwargs['data'].get('dataset')
    seq_len = kwargs['model'].get('seq_len')
    horizon = kwargs['model'].get('horizon')

    time = np.load(data_npz)['time']
    # horizon is in seq_len. the last
    T = len(time) - seq_len - horizon

    lon = np.load(data_npz)['output_lon']
    lat = np.load(data_npz)['output_lat']
    precip = np.load(data_npz)['output_precip']

    input_conv2d_gsmap = np.zeros(shape=(T, seq_len, len(lat), len(lon), 1))
    target_conv2d_gsmap = np.zeros(shape=(T, horizon, len(lat), len(lon), 1))

    gauge_dataset = kwargs['data'].get('gauge_dataset')
    gauge_lon = np.load(gauge_dataset)['gauge_lon']
    gauge_lat = np.load(gauge_dataset)['gauge_lat']
    gauge_precipitation = np.load(gauge_dataset)['gauge_precip']
    for i in range(len(gauge_lat)):
        lat = gauge_lat[i]
        lon = gauge_lon[i]
        temp_lat = int(round((23.95 - lat) / 0.1))
        temp_lon = int(round((lon - 100.05) / 0.1))
        gauge_precip = gauge_precipitation[:, i]
        for j in range(T):
            input_conv2d_gsmap[j, :, temp_lat, temp_lon,
                               1] = gauge_precip[j:j + seq_len]
            # remap the target gsmap by gauge data
            target_conv2d_gsmap[j, :, temp_lat, temp_lon,
                                0] = gauge_precip[j + seq_len:j + seq_len +
                                                  horizon]
    return input_conv2d_gsmap, target_conv2d_gsmap


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
