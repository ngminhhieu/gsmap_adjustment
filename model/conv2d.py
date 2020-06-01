from keras.layers import Conv2D, BatchNormalization, MaxPooling2D, UpSampling2D
from keras.models import Sequential
import numpy as np
from model import common_util
import model.utils.conv2d as utils_conv2d
import os
import yaml
from pandas import read_csv
from keras.utils import plot_model
from keras import backend as K
from keras.losses import mse


class Conv2DSupervisor():
    def __init__(self, **kwargs):
        self.config_model = common_util.get_config_model(**kwargs)

        # load_data
        self.data = utils_conv2d.load_dataset(**kwargs)
        self.input_train = self.data['input_train']
        self.input_valid = self.data['input_valid']
        self.input_test = self.data['input_test']
        self.target_train = self.data['target_train']
        self.target_valid = self.data['target_valid']
        self.target_test = self.data['target_test']

        # other configs
        self.log_dir = self.config_model['log_dir']
        self.optimizer = self.config_model['optimizer']
        self.loss = self.config_model['loss']
        self.activation = self.config_model['activation']
        self.batch_size = self.config_model['batch_size']
        self.epochs = self.config_model['epochs']
        self.callbacks = self.config_model['callbacks']
        self.seq_len = self.config_model['seq_len']
        self.horizon = self.config_model['horizon']

        self.model = self.build_model_prediction()

    def build_model_prediction(self):
        model = Sequential()

        # Input
        model.add(
            Conv2D(filters=32,
                   kernel_size=(3, 3),
                   padding='same',
                   activation=self.activation,
                   name='input_layer_conv2d',
                   input_shape=(160, 120, 1)))
        model.add(BatchNormalization())

        # Max Pooling - Go deeper
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(
            Conv2D(32, (3, 3),
                   activation='relu',
                   padding='same',
                   name='hidden_conv2d_1'))
        model.add(BatchNormalization())
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(
            Conv2D(32, (3, 3),
                   activation='relu',
                   padding='same',
                   name='hidden_conv2d_2'))
        model.add(BatchNormalization())
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(
            Conv2D(64, (3, 3),
                   activation='relu',
                   padding='same',
                   name='hidden_conv2d_3'))
        model.add(BatchNormalization())

        # Up Sampling
        model.add(UpSampling2D(size=(2, 2)))
        model.add(
            Conv2D(64, (3, 3),
                   activation='relu',
                   padding='same',
                   name='hidden_conv2d_4'))
        model.add(BatchNormalization())
        model.add(UpSampling2D(size=(2, 2)))
        model.add(
            Conv2D(32, (3, 3),
                   activation='relu',
                   padding='same',
                   name='hidden_conv2d_5'))
        model.add(BatchNormalization())
        model.add(UpSampling2D(size=(2, 2)))
        model.add(
            Conv2D(32, (3, 3),
                   activation='relu',
                   padding='same',
                   name='hidden_conv2d_6'))
        model.add(BatchNormalization())

        model.add(
            Conv2D(filters=1,
                   kernel_size=(3, 3),
                   padding='same',
                   name='output_layer_conv2d',
                   activation=self.activation))
        print(model.summary())

        # plot model
        from keras.utils import plot_model
        plot_model(model=model,
                   to_file=self.log_dir + '/conv2d_model.png',
                   show_shapes=True)
        return model

    def train(self):
        self.model.compile(optimizer=self.optimizer,
                           loss=self.loss,
                           metrics=['mse', 'mae'])

        training_history = self.model.fit(self.input_train,
                                          self.target_train,
                                          batch_size=self.batch_size,
                                          epochs=self.epochs,
                                          callbacks=self.callbacks,
                                          validation_data=(self.input_valid,
                                                           self.target_valid),
                                          shuffle=True,
                                          verbose=1)

        if training_history is not None:
            common_util._plot_training_history(training_history,
                                               self.config_model)
            common_util._save_model_history(training_history,
                                            self.config_model)
            config = dict(self.config_model['kwargs'])

            # create config file in log again
            config_filename = 'config.yaml'
            config['train']['log_dir'] = self.log_dir
            with open(os.path.join(self.log_dir, config_filename), 'w') as f:
                yaml.dump(config, f, default_flow_style=False)

    def test_prediction(self):
        print("Load model from: {}".format(self.log_dir))
        self.model.load_weights(self.log_dir + 'best_model.hdf5')
        self.model.compile(optimizer=self.optimizer, loss=self.loss)

        input_test = self.input_test
        actual_data = self.target_test
        predicted_data = np.zeros(shape=(len(actual_data), 160, 120, 1))
        from tqdm import tqdm
        iterator = tqdm(range(0, len(actual_data)))
        for i in iterator:
            input = np.zeros(shape=(1, 160, 120, 1))
            input[0] = input_test[i].copy()
            yhats = self.model.predict(input)
            predicted_data[i] = yhats[0]

        dataset = self.config_model['data_kwargs'].get('dataset')
        gauge_lon = np.load(dataset)['gauge_lon']
        gauge_lat = np.load(dataset)['gauge_lat']

        groundtruth = []
        preds = []
        num_gt = 0
        num_preds = 0
        list_metrics = np.zeros(shape=(len(gauge_lat) + 1, 3))
        # MAE for only gauge data
        for i in range(len(gauge_lat)):
            lat = gauge_lat[i]
            lon = gauge_lon[i]
            temp_lat = int(round((23.95 - lat) / 0.1))
            temp_lon = int(round((lon - 100.05) / 0.1))

            # gauge data
            gt = actual_data[:, temp_lat, temp_lon, 0].copy()
            groundtruth.append(gt)

            # prediction data
            yhat = predicted_data[:, temp_lat, temp_lon, 0].copy()
            preds.append(yhat)

            x = np.count_nonzero(yhat > 0)
            y = np.count_nonzero(gt > 0)

            list_metrics[i, 0] = common_util.mae(gt, yhat)
            list_metrics[i, 1] = common_util.rmse(gt, yhat)
            margin = y - x
            total_margin = total_margin + math.abs(margin)
            list_metrics[i, 2] = margin

        # total of 72 gauges
        list_metrics[0, 0] = common_util.mae(groundtruth, preds)
        list_metrics[0, 1] = common_util.rmse(groundtruth, preds)
        list_metrics[0, 2] = total

        groundtruth = np.array(groundtruth)
        preds = np.array(preds)
        np.savetxt(self.log_dir + 'groundtruth.csv', groundtruth, delimiter=",")
        np.savetxt(self.log_dir + 'preds.csv', preds, delimiter=",")

    def plot_result(self):
        from matplotlib import pyplot as plt
        preds = np.load(self.log_dir + 'pd.npy')
        gt = np.load(self.log_dir + 'gt.npy')
        plt.plot(preds[:], label='preds')
        plt.plot(gt[:], label='gt')
        plt.legend()
        plt.savefig(self.log_dir + 'result_predict.png')
        plt.close()
