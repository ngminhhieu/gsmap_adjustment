from keras.models import Sequential
from keras.layers import Flatten, Dense, MaxPooling2D, Conv2D, Conv2DTranspose, UpSampling2D, Cropping2D, Cropping3D, MaxPooling3D
from keras.layers.convolutional import Conv3D, Conv3DTranspose
from keras.layers.convolutional_recurrent import ConvLSTM2D
from keras.layers.normalization import BatchNormalization
import numpy as np
import pylab as plt
from model import common_util
import model.utils.conv2d as utils_conv2d
import os
import yaml
from pandas import read_csv


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

        self.model = self.build_model()

    def build_model(self):
        model = Sequential()

        # extract useful information
        model.add(
            ConvLSTM2D(
                filters=32,
                kernel_size=(3, 3),
                padding='same',
                #    activation=self.activation,
                return_sequences=True,
                input_shape=(self.seq_len, 72, 72, 3)))

        model.add(BatchNormalization())

        model.add(
            ConvLSTM2D(filters=32,
                       kernel_size=(3, 3),
                       padding='same',
                       return_sequences=True))
        model.add(BatchNormalization())

        model.add(Conv3D(filters=32, kernel_size=(3, 3, 3), padding='same'))
        model.add(MaxPooling3D(pool_size=(1, 2, 2)))

        # scaling up
        model.add(
            Conv3DTranspose(filters=32,
                            kernel_size=(3, 3, 3),
                            strides=(1, 5, 4)))

        # ((top_crop, bottom_crop), (left_crop, right_crop))
        model.add(Cropping3D(cropping=((4, 4), (10, 10), (12, 12))))

        model.add(Conv3D(filters=3, kernel_size=(3, 3, 3), padding='same', activaiton = 'sigmoid'))

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
                                          verbose=2)

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

    def test(self):
        print("Load model from: {}".format(self.log_dir))
        self.model.load_weights(self.log_dir + 'best_model.hdf5')
        self.model.compile(optimizer=self.optimizer, loss=self.loss)

        input_test = self.input_test
        input_train = self.input_train
        actual_data = self.target_test
        # predicted_data = np.zeros(shape=(len(actual_data), self.horizon, 160,
        #                                  120, 3))
        predicted_data = np.zeros(shape=(len(self.target_train), self.seq_len,
                                         72, 72, 3))
        from tqdm import tqdm
        iterator = tqdm(
            range(0,
                  len(self.target_train) - self.seq_len - self.horizon,
                  self.horizon))
        for i in iterator:
            input = np.zeros(shape=(1, self.seq_len, 72, 72, 3))
            input[0, :, :, :, :] = input_train[i].copy()
            # input = input[np.newaxis, :, :, :, :]
            predicted_data[i] = self.model.predict(input)

        print(predicted_data[predicted_data[:, :, :, :, 2] > 0])
        print(predicted_data)
        np.save(self.log_dir + 'pd', predicted_data)

        actual_data = actual_data.flatten()
        predicted_data = predicted_data.flatten()

        common_util.mae(actual_data, predicted_data)
        common_util.mse(actual_data, predicted_data)
        common_util.rmse(actual_data, predicted_data)

    def check(self):
        print("Load model from: {}".format(self.log_dir))
        self.model.load_weights(self.log_dir + 'best_model.hdf5')
        self.model.compile(optimizer=self.optimizer, loss=self.loss)
        input_test = self.input_test
        actual_data = self.target_test
        predicted_data = np.zeros(shape=(len(input_test), 160, 120, 3))

        for i in range(0, len(input_test)):
            input = input_test[i].copy()
            input = input.reshape(1, 72, 72, 3)
            predicted_data[i] = self.model.predict(input)

        # total_mae = 0
        actual_arr = []
        preds_arr = []
        for lat in range(72):
            for lon in range(72):
                os.system(
                    'cdo -outputtab,value -remapnn,lon={}_lat={} data/conv2d_gsmap/gsmap_2011_2018.nc > data/test/precip.csv'
                    .format(input_test[-1, 0, lon, 1], input_test[-1, lat, 0,
                                                                  0]))
            precipitation = read_csv('data/test/precip.csv')
            actual = precipitation.to_numpy()
            actual = actual[-354:, 0]
            print(actual)
            actual_arr.append(actual)
            preds = predicted_data[:, lat, lon, 2]
            print(preds)
            preds_arr.append(preds)

        common_util.mae(actual_arr.flatten(), preds_arr.flatten())

    def plot_result(self):
        from matplotlib import pyplot as plt
        preds = np.load(self.log_dir + 'pd.npy')
        gt = np.load(self.log_dir + 'gt.npy')
        plt.plot(preds[:], label='preds')
        plt.plot(gt[:], label='gt')
        plt.legend()
        plt.savefig(self.log_dir + 'result_predict.png')
        plt.close()
