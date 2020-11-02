import numpy as np
from model import common_util
from sklearn.preprocessing import MinMaxScaler
from keras import backend as K
import pandas as pd

def create_data_prediction(**kwargs):

    # data_npz = kwargs['data'].get('dataset')
    # dataset = np.load(data_npz)['dataset']
    dataset_gsmap = pd.read_csv('data/ann/gsmap.csv').to_numpy()
    dataset_gauge = pd.read_csv('data/ann/gauge.csv').to_numpy()
    T = len(dataset_gsmap)
    
    input_model = np.zeros(shape=(T*dataset_gsmap.shape[1], 1))
    output_model = np.zeros(shape=(T*dataset_gsmap.shape[1], 1))

    for col in range(dataset_gsmap.shape[1]):
        for row in range(T):
            input_model[col*T + row] = dataset_gsmap[row, col]
            output_model[col*T + row] = dataset_gauge[row, col]

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