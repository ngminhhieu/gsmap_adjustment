import numpy as np
from model import common_util
from sklearn.preprocessing import MinMaxScaler
from keras import backend as K
import pandas as pd

def create_data_prediction(dataset_gsmap, dataset_gauge, **kwargs):

    # data_npz = kwargs['data'].get('dataset')
    # dataset = np.load(data_npz)['dataset']
    input_dim = kwargs['model'].get('input_dim')
    output_dim = kwargs['model'].get('output_dim')
    seq_len = kwargs['model'].get('seq_len')
    horizon = kwargs['model'].get('horizon')
    T = len(dataset_gsmap)
    col = dataset_gsmap.shape[1]
    input_model = np.zeros(shape=(T*col, seq_len, input_dim))
    output_model = np.zeros(shape=(T*col, output_dim))

    for col in range(dataset_gsmap.shape[1]):
        for row in range(T-seq_len-horizon):
            input_model[col*T + row, :, 0] = dataset_gsmap[row:row+seq_len, col]
            output_model[col*T + row, 0] = dataset_gauge[row+seq_len, col]

    return input_model, output_model


def load_dataset(**kwargs):
    dataset_gsmap = pd.read_csv('data/ann/gsmap.csv').to_numpy()
    dataset_gauge = pd.read_csv('data/ann/gauge.csv').to_numpy()
    dataset_gsmap = dataset_gsmap[:, 2]
    dataset_gauge = dataset_gauge[:, 2]
    dataset_gsmap = np.reshape(dataset_gsmap, (dataset_gsmap.shape[0], 1))
    dataset_gauge = np.reshape(dataset_gauge, (dataset_gauge.shape[0], 1))
    scaler = MinMaxScaler(copy=True, feature_range=(0, 1))
    scaler.fit(dataset_gsmap)
    dataset_gsmap = scaler.transform(dataset_gsmap)
    dataset_gauge = scaler.transform(dataset_gauge)

    input_lstm, target_lstm = create_data_prediction(dataset_gsmap, dataset_gauge, **kwargs)
    # input_lstm = scaler.transform(input_lstm)
    # target_lstm = scaler.transform(target_lstm)
    # get test_size, valid_size from config
    test_size = kwargs['data'].get('test_size')
    valid_size = kwargs['data'].get('valid_size')

    # split data to train_set, valid_set, test_size
    input_train, input_valid, input_test = common_util.prepare_train_valid_test(
        input_lstm, test_size=test_size, valid_size=valid_size)
    target_train, target_valid, target_test = common_util.prepare_train_valid_test(
        target_lstm, test_size=test_size, valid_size=valid_size)
        
    data = {}
    for cat in ["train", "valid", "test"]:
        x, y = locals()["input_" + cat], locals()["target_" + cat]
        data["input_" + cat] = x
        data["target_" + cat] = y

    data["scaler"] = scaler
    return data