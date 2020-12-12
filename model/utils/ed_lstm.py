import numpy as np
from model import common_util
from sklearn.preprocessing import MinMaxScaler
from keras import backend as K
import pandas as pd


def create_data_prediction_overlap_all(dataset_gsmap, wind_u_mean, wind_v_mean,
                                       surface_temp, precip_seasonal,
                                       precip_trend, dataset_gauge, **kwargs):

    input_dim = kwargs['model'].get('input_dim')
    output_dim = kwargs['model'].get('output_dim')
    seq_len = kwargs['model'].get('seq_len')
    horizon = kwargs['model'].get('horizon')
    T = len(dataset_gsmap)
    col = dataset_gsmap.shape[1]
    input_encoder = np.empty(shape=((T - seq_len - horizon) * col, seq_len, input_dim))
    input_decoder = np.empty(shape=((T - seq_len - horizon) * col, seq_len, output_dim))
    output_decoder = np.empty(shape=((T - seq_len - horizon) * col, seq_len, output_dim))

    for col in range(dataset_gsmap.shape[1]):
        for row in range(T - seq_len - horizon):
            input_encoder[col * T + row, :,
                          0] = dataset_gsmap[row + horizon:row + seq_len +
                                             horizon, col].copy()
            input_encoder[col * T + row, :,
                          1] = wind_u_mean[row + horizon:row + seq_len +
                                           horizon, col].copy()
            input_encoder[col * T + row, :,
                          2] = wind_v_mean[row + horizon:row + seq_len +
                                           horizon, col].copy()
            # input_encoder[col * T + row, :,
            #               3] = surface_temp[row + horizon:row + seq_len +
            #                                 horizon, col].copy()
            # input_encoder[col * T + row, :,
            #               3] = precip_seasonal[row + horizon:row + seq_len +
            #                                 horizon, col].copy()
            # input_encoder[col * T + row, :,
            #               4] = precip_trend[row + horizon:row + seq_len +
            #                                 horizon, col].copy()
            input_decoder[col * T + row, :,
                          0] = dataset_gauge[row + horizon - 1:row + seq_len +
                                             horizon - 1, col].copy()
            input_decoder[col * T + row, 0, 0] = 0
            output_decoder[col * T + row, :,
                           0] = dataset_gauge[row + horizon:row + seq_len +
                                              horizon, col].copy()

    return input_encoder, input_decoder, output_decoder


def create_data_prediction_all(dataset_gsmap, dataset_gauge, **kwargs):

    input_dim = kwargs['model'].get('input_dim')
    output_dim = kwargs['model'].get('output_dim')
    seq_len = kwargs['model'].get('seq_len')
    horizon = kwargs['model'].get('horizon')
    T = len(dataset_gsmap)
    input_encoder = np.zeros(shape=(T, seq_len, input_dim))
    input_decoder = np.zeros(shape=(T, seq_len, input_dim))
    output_decoder = np.zeros(shape=(T, seq_len, output_dim))

    for row in range(T - seq_len - horizon):
        input_encoder[row, :, :] = dataset_gsmap[row + horizon:row + seq_len +
                                                 horizon].copy()
        input_decoder[row, :, :] = dataset_gauge[row + horizon - 1:row +
                                                 seq_len + horizon - 1].copy()
        input_decoder[row, 0, :] = 0
        output_decoder[row, :, :] = dataset_gauge[row + horizon:row + seq_len +
                                                  horizon].copy()

    return input_encoder, input_decoder, output_decoder


def load_dataset(**kwargs):
    dataset_gsmap = pd.read_csv('data/ann/gsmap.csv', header=None).to_numpy()
    dataset_gsmap = np.reshape(np.ravel(dataset_gsmap, order='F'), (-1,1))
    wind_u_mean = pd.read_csv('data/ann/wind_u_mean.csv', header=None).to_numpy()
    wind_u_mean = np.reshape(np.ravel(wind_u_mean, order='F'), (-1,1))
    wind_v_mean = pd.read_csv('data/ann/wind_v_mean.csv', header=None).to_numpy()
    wind_v_mean = np.reshape(np.ravel(wind_v_mean, order='F'), (-1,1))
    surface_temp = pd.read_csv('data/ann/surface_temp.csv', header=None).to_numpy()
    surface_temp = np.reshape(np.ravel(surface_temp, order='F'), (-1,1))
    precip_trend = pd.read_csv('data/ann/precip_trend.csv', header=None).to_numpy()
    precip_trend = np.reshape(np.ravel(precip_trend, order='F'), (-1,1))
    precip_seasonal = pd.read_csv('data/ann/precip_seasonal.csv', header=None).to_numpy()
    precip_seasonal = np.reshape(np.ravel(precip_seasonal, order='F'), (-1,1))
    dataset_gauge = pd.read_csv('data/ann/gauge.csv', header=None).to_numpy()
    dataset_gauge = np.reshape(np.ravel(dataset_gauge, order='F'), (-1,1))
    # dataset_gsmap = dataset_gsmap[:, 0]
    # dataset_gauge = dataset_gauge[:, 0]

    # dataset_gsmap = np.reshape(dataset_gsmap, (dataset_gsmap.shape[0], 1))
    # dataset_gauge = np.reshape(dataset_gauge, (dataset_gauge.shape[0], 1))
    # dataset_gsmap = np.tile(dataset_gsmap, (1,72))
    # dataset_gauge = np.tile(dataset_gauge, (1,72))
    scaler = MinMaxScaler(copy=True, feature_range=(0, 1))
    scaler.fit(dataset_gsmap)
    dataset_gsmap = scaler.transform(dataset_gsmap)
    dataset_gauge = scaler.transform(dataset_gauge)
    wind_u_mean = scaler.transform(wind_u_mean)
    wind_v_mean = scaler.transform(wind_v_mean)
    surface_temp = scaler.transform(surface_temp)
    precip_seasonal = scaler.transform(precip_seasonal)
    precip_trend = scaler.transform(precip_trend)

    input_encoder, input_decoder, target_decoder = create_data_prediction_overlap_all(
        dataset_gsmap, wind_u_mean, wind_v_mean, surface_temp, precip_seasonal,
        precip_trend, dataset_gauge, **kwargs)
    test_size = kwargs['data'].get('test_size')
    valid_size = kwargs['data'].get('valid_size')
    train_size = 1 - test_size - valid_size

    # split data to train_set, valid_set, test_size
    my_dataset = pd.read_csv('data/ann/gsmap.csv', header=None).to_numpy()
    input_encoder_train = np.empty(shape=(int(input_encoder.shape[0]*train_size), input_encoder.shape[1], input_encoder.shape[2]))
    input_encoder_valid = np.empty(shape=(int(input_encoder.shape[0]*valid_size), input_encoder.shape[1], input_encoder.shape[2]))
    input_encoder_test = np.empty(shape=(int(input_encoder.shape[0]*test_size), input_encoder.shape[1], input_encoder.shape[2]))
    input_decoder_train = np.empty(shape=(int(input_decoder.shape[0]*train_size), input_decoder.shape[1], input_decoder.shape[2]))
    input_decoder_valid = np.empty(shape=(int(input_decoder.shape[0]*valid_size), input_decoder.shape[1], input_decoder.shape[2]))
    input_decoder_test = np.empty(shape=(int(input_decoder.shape[0]*test_size), input_decoder.shape[1], input_decoder.shape[2]))
    target_decoder_train = np.empty(shape=(int(target_decoder.shape[0]*train_size), target_decoder.shape[1], target_decoder.shape[2]))
    target_decoder_valid = np.empty(shape=(int(target_decoder.shape[0]*valid_size), target_decoder.shape[1], target_decoder.shape[2]))
    target_decoder_test = np.empty(shape=(int(target_decoder.shape[0]*test_size), target_decoder.shape[1], target_decoder.shape[2]))
    gauges = my_dataset.shape[1]
    for i in range(gauges):
        train_pivot = int(len(my_dataset)*train_size)
        valid_pivot = int(len(my_dataset)*(train_size+valid_size))
        test_pivot = int(len(my_dataset)*(train_size+valid_size+test_size))
        input_encoder_train[i*train_size:(i+1)*train_size] = input_encoder[i*gauges:i*gauges+train_pivot]
        input_encoder_valid[i*train_size:(i+1)*train_size] = input_encoder[i*gauges+train_pivot:i*gauges+train_pivot+valid_pivot]
        input_encoder_test[i*train_size:(i+1)*train_size] = input_encoder[i*gauges+train_pivot+valid_pivot:i*gauges+train_pivot+valid_pivot+test_pivot]
        input_decoder_train[i*train_size:(i+1)*train_size] = input_decoder[i*gauges:i*gauges+train_pivot]
        input_decoder_valid[i*train_size:(i+1)*train_size] = input_decoder[i*gauges+train_pivot:i*gauges+train_pivot+valid_pivot]
        input_decoder_test[i*train_size:(i+1)*train_size] = input_decoder[i*gauges+train_pivot+valid_pivot:i*gauges+train_pivot+valid_pivot+test_pivot]
        target_decoder_train[i*train_size:(i+1)*train_size] = target_decoder[i*gauges:i*gauges+train_pivot]
        target_decoder_valid[i*train_size:(i+1)*train_size] = target_decoder[i*gauges+train_pivot:i*gauges+train_pivot+valid_pivot]
        target_decoder_test[i*train_size:(i+1)*train_size] = target_decoder[i*gauges+train_pivot+valid_pivot:i*gauges+train_pivot+valid_pivot+test_pivot]

    # input_encoder_train, input_encoder_valid, input_encoder_test = common_util.prepare_train_valid_test(
    #     input_encoder, test_size=test_size, valid_size=valid_size)
    # input_decoder_train, input_decoder_valid, input_decoder_test = common_util.prepare_train_valid_test(
    #     input_decoder, test_size=test_size, valid_size=valid_size)
    # target_decoder_train, target_decoder_valid, target_decoder_test = common_util.prepare_train_valid_test(
    #     target_decoder, test_size=test_size, valid_size=valid_size)

    data = {}
    for cat in ["train", "valid", "test"]:
        e_x, e_y, d_y = locals()["input_encoder_" + cat], locals()[
            "input_decoder_" + cat], locals()["target_decoder_" + cat]
        data["input_encoder_" + cat] = e_x
        data["input_decoder_" + cat] = e_y
        data["target_decoder_" + cat] = d_y

    data["scaler"] = scaler
    return data