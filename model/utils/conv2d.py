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
    # map_precip = np.load(data_npz)['map_precip']
    map_precip_all = np.load('../drive/My Drive/precip_adjustment_data/min_all.npz')['map_precip']
    map_precip_r05 = np.load('../drive/My Drive/precip_adjustment_data/min_r05.npz')['map_precip']

    # gauge_lon = np.load(data_npz)['gauge_lon']
    # gauge_lat = np.load(data_npz)['gauge_lat']
    gauge_precip = np.load(data_npz)['gauge_precip']

    # input is gsmap
    input_model = np.zeros(shape=(T, 160, 120, 2))
    # output is gauge
    output_model = np.zeros(shape=(T, 160, 120, 1))


    for i in range(len(map_lat)):
        lat = map_lat[i]
        lon = map_lon[i]
        temp_lat = int(round((23.95 - lat) / 0.1))
        temp_lon = int(round((lon - 100.05) / 0.1))
        input_model[:, temp_lat, temp_lon, 0] = map_precip_all[:, i]
        output_model[:, temp_lat, temp_lon, 0] = gauge_precip[:, i]
        input_model[:, temp_lat, temp_lon, 1] = map_precip_r05[:, i]
        # output_model[:, temp_lat, temp_lon, 1] = gauge_precip[:, i]
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