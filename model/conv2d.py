from keras.models import Sequential
from keras.layers import Flatten, Dense, MaxPooling2D, Conv2D, Conv2DTranspose, UpSampling2D, Cropping2D, Cropping3D, MaxPooling3D, UpSampling3D
from keras.layers.convolutional import Conv3D, Conv3DTranspose
from keras.layers.convolutional_recurrent import ConvLSTM2D
from keras.layers.normalization import BatchNormalization
import numpy as np
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

        self.model = self.build_model_prediction()
    
    def build_model_generation(self):
        model = Sequential()

        # extract useful information
        model.add(
            ConvLSTM2D(filters=16,
                       kernel_size=(3, 3),
                       padding='same',
                       return_sequences=True,
                       activation = self.activation,
                       name = 'input_layer_convlstm2d',
                       # set seq_len is None to make the output flexible
                       input_shape=(None, 160, 120, 1)))
        model.add(BatchNormalization())

        model.add(
            ConvLSTM2D(filters=16,
                       kernel_size=(3, 3),
                       padding='same',
                       activation = self.activation,
                    #    name='hidden_layer_convlstm2d_1',
                       return_sequences=True))
        model.add(BatchNormalization())

        model.add(
            ConvLSTM2D(filters=16,
                       kernel_size=(3, 3),
                       padding='same',
                       activation = self.activation,
                    #    name='hidden_layer_convlstm2d_2',
                       return_sequences=True))
        model.add(BatchNormalization())

        model.add(
            ConvLSTM2D(filters=16,
                       kernel_size=(3, 3),
                       padding='same',
                       activation = self.activation,
                    #    name='hidden_layer_convlstm2d_3',
                       return_sequences=True))
        model.add(BatchNormalization())

        # model.add(Conv3D(filters=32, kernel_size=(3, 3, 3),
        # activation = self.activation, 
        # padding='same'))
        # model.add(BatchNormalization())
        
        # model.add(MaxPooling3D(pool_size=(1, 2, 2)))

        # # scaling up
        # model.add(
        #     Conv3DTranspose(filters=32,
        #                     kernel_size=(3, 3, 3),
        #                     activation = self.activation,
        #                     strides=(1, 5, 4)))
        # model.add(BatchNormalization())

        # # # ((top_crop, bottom_crop), (left_crop, right_crop))
        # model.add(Cropping3D(cropping=((4, 4), (10, 10), (12, 12))))
        # model.add(Cropping3D(cropping=((7, 7), (0, 0), (0, 0)), name='cropping_layer'))

        model.add(
            Conv3D(filters=1,
                   kernel_size=(3, 3, 3),
                   padding='same',
                   name='output_layer_conv3d',
                   activation=self.activation))

        print(model.summary())

        # plot model
        from keras.utils import plot_model
        plot_model(model=model,
                   to_file=self.log_dir + '/conv2d_model.png',
                   show_shapes=True)
        return model

    def build_model_prediction(self):
        model = Sequential()
        model.add(
            ConvLSTM2D(filters=16,
                       kernel_size=(3, 3),
                       padding='same',
                       return_sequences=True,
                       activation = self.activation,
                       name = 'input_layer_convlstm2d',
                       input_shape=(self.seq_len, 160, 120, 1)))
        model.add(BatchNormalization())

        model.add(MaxPooling3D(pool_size=(2, 2, 2)))

        model.add(
            ConvLSTM2D(filters=32,
                       kernel_size=(3, 3),
                       padding='same',
                       activation = self.activation,
                       name='hidden_layer_convlstm2d_1',
                       return_sequences=True))
        model.add(BatchNormalization())

        model.add(MaxPooling3D(pool_size=(2, 2, 2)))

        model.add(
            ConvLSTM2D(filters=64,
                       kernel_size=(3, 3),
                       padding='same',
                       activation = self.activation,
                       name='hidden_layer_convlstm2d_2',
                       return_sequences=True))
        model.add(BatchNormalization())

        model.add(
            ConvLSTM2D(filters=64,
                       kernel_size=(3, 3),
                       padding='same',
                       activation = self.activation,
                       name='hidden_layer_convlstm2d_3',
                       return_sequences=True))
        model.add(BatchNormalization())

        # Upsampling
        model.add(UpSampling3D(size=(2, 2, 2)))

        model.add(
            ConvLSTM2D(filters=32,
                       kernel_size=(3, 3),
                       padding='same',
                       activation = self.activation,
                       name='hidden_layer_convlstm2d_4',
                       return_sequences=True))
        model.add(BatchNormalization())

        model.add(UpSampling3D(size=(2, 2, 2)))

        model.add(
            ConvLSTM2D(filters=16,
                       kernel_size=(3, 3),
                       padding='same',
                       activation = self.activation,
                       name='hidden_layer_convlstm2d_5',
                       return_sequences=True))
        model.add(BatchNormalization())

        model.add(
            ConvLSTM2D(filters=8,
                       kernel_size=(3, 3),
                       padding='same',
                       activation = self.activation,
                       name='hidden_layer_convlstm2d_6',
                       return_sequences=True))
        model.add(BatchNormalization())

        model.add(
            Conv3D(filters=1,
                   kernel_size=(3, 3, 3),
                   padding='same',
                   name='output_layer_conv3d',
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

    def test_prediction(self):
        import sys
        print("Load model from: {}".format(self.log_dir))
        self.model.load_weights(self.log_dir + 'best_model.hdf5')
        self.model.compile(optimizer=self.optimizer, loss=self.loss)

        input_test = self.input_test
        actual_data = self.target_test
        predicted_data = np.zeros(shape=(len(actual_data), self.horizon, 160,
                                         120, 1))
        from tqdm import tqdm
        iterator = tqdm(range(0,len(actual_data)))
        for i in iterator:
            input = np.zeros(shape=(1, self.seq_len, 160, 120, 1))
            input[0] = input_test[i].copy()
            yhats = self.model.predict(input)   
            predicted_data[i, 0] = yhats[0, -1]            
            print("Prediction: ", np.count_nonzero(predicted_data[i, 0] > 0), "Actual: ", np.count_nonzero(actual_data[i, -1]>0))
        
        data_npz = self.config_model['data_kwargs'].get('dataset')
        lon = np.load(data_npz)['input_lon']
        lat = np.load(data_npz)['input_lat']

        gauge_dataset = self.config_model['data_kwargs'].get('gauge_dataset')
        gauge_lon = np.load(gauge_dataset)['gauge_lon']
        gauge_lat = np.load(gauge_dataset)['gauge_lat']
        gauge_precipitation = np.load(gauge_dataset)['gauge_precip']

        actual_arr = []
        preds_arr = []
        gauge_arr = []

        # MAE for 72x72 matrix including gauge data
        for index, lat in np.ndenumerate(lat):
            temp_lat = int(round((23.95-lat)/0.1))

            for index, lon in np.ndenumerate(lon):
                temp_lon = int(round((lon-100.05)/0.1))

                # actual data
                actual_precip = actual_data[:, 0, temp_lat, temp_lon, 0]
                actual_arr.append(actual_precip)

                # prediction data
                preds = predicted_data[:, 0, temp_lat, temp_lon, 0]
                preds_arr.append(preds)
        
        common_util.mae(actual_arr, preds_arr)
        
        preds_arr = []
        # MAE for only gauge data
        for i in range(len(gauge_lat)):
            lat = gauge_lat[i]
            lon = gauge_lon[i]
            temp_lat = int(round((23.95-lat)/0.1))
            temp_lon = int(round((lon-100.05)/0.1))

            # gauge data
            gauge_precip = gauge_precipitation[-353:, i]
            gauge_arr.append(gauge_precip)

            # prediction data
            preds = predicted_data[:, 0, temp_lat, temp_lon, 0]
            preds_arr.append(preds)

        common_util.cal_error(gauge_arr, preds_arr)

    def test_generation(self):
        for lat_index in range(72):
            lat = input_test[-1, -1, lat_index, 0, 0]
            
            for lon_index in range(72):
                lon = input_test[-1, -1, 0, lon_index, 1]
                temp_lat = int(round((23.95-lat)/0.1))
                temp_lon = int(round((lon-100.05)/0.1))
                # print(lat, lon)
                # print(temp_lat, temp_lon)
                # print(actual_data[-1, -1, temp_lat, 0, 0], actual_data[-1, -1, 0, temp_lon, 1])
                actual_precip = actual_data[:, 0, temp_lat, temp_lon, 0]
                # actual_precip = actual_data[:, 0, lat_index, lon_index, 2]
                actual_arr.append(actual_precip)
                print(actual_precip)
                # preds = predicted_data[:, 0, lat_index, lon_index, 2]
                preds = predicted_data[:, 0, temp_lat, temp_lon, 0]
                print(preds)
                preds_arr.append(preds)

        common_util.mae(actual_arr, preds_arr)

    def plot_result(self):
        from matplotlib import pyplot as plt
        preds = np.load(self.log_dir + 'pd.npy')
        gt = np.load(self.log_dir + 'gt.npy')
        plt.plot(preds[:], label='preds')
        plt.plot(gt[:], label='gt')
        plt.legend()
        plt.savefig(self.log_dir + 'result_predict.png')
        plt.close()
