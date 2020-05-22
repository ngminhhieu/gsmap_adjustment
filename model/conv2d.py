from keras.layers import Dense, Input
from keras.layers import Conv2D, Flatten, Lambda
from keras.layers import Reshape, Conv2DTranspose
from keras.models import Model
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

        self.vae, self.vae_enc_dec = self.build_model_prediction()

    def build_model_prediction(self):
<<<<<<< HEAD
        model = Sequential()

        # Input
        model.add(
            ConvLSTM2D(filters=16,
                       kernel_size=(3, 3),
                       padding='same',
                       return_sequences=True,
                       activation=self.activation,
                       name='input_layer_convlstm2d',
                       input_shape=(self.seq_len, 160, 120, 1)))
        model.add(BatchNormalization())

        # Max Pooling - Go deeper
        model.add(MaxPooling3D(pool_size=(2, 2, 1)))

        model.add(
            ConvLSTM2D(filters=32,
                       kernel_size=(3, 3),
                       padding='same',
                       activation=self.activation,
                       name='hidden_layer_convlstm2d_1',
                       return_sequences=True))
        model.add(BatchNormalization())

        model.add(MaxPooling3D(pool_size=(2, 2, 1)))

        model.add(
            ConvLSTM2D(filters=64,
                       kernel_size=(3, 3),
                       padding='same',
                       activation=self.activation,
                       name='hidden_layer_convlstm2d_2',
                       return_sequences=True))
        model.add(BatchNormalization())

        # Up Sampling
        model.add(UpSampling3D(size=(2, 2, 1)))

        model.add(
            ConvLSTM2D(filters=32,
                       kernel_size=(3, 3),
                       padding='same',
                       activation=self.activation,
                       name='hidden_layer_convlstm2d_3',
                       return_sequences=True))
        model.add(BatchNormalization())

        model.add(UpSampling3D(size=(2, 2, 1)))

        model.add(
            ConvLSTM2D(filters=16,
                       kernel_size=(3, 3),
                       padding='same',
                       activation=self.activation,
                       name='hidden_layer_convlstm2d_4',
                       return_sequences=True))
        model.add(BatchNormalization())

        # model.add(Cropping3D(cropping=((3,0), (0,0), (0,0))))

        model.add(
            Conv3D(filters=1,
                   kernel_size=(3, 3, 1),
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
=======
        input_shape=(160, 120, 1)
        kernel_size = 3
        latent_dim = 2
        filters = 16

        inputs = Input(shape=input_shape, name='encoder_input')
        x = inputs
        for i in range(2):
            filters *= 2
            x = Conv2D(filters=filters,
                    kernel_size=kernel_size,
                    activation=self.activation,
                    strides=2,
                    padding='same')(x)

        # shape info needed to build decoder model
        shape = K.int_shape(x)

        # generate latent vector Q(z|X)
        x = Flatten()(x)
        x = Dense(16, activation='relu')(x)
        z_mean = Dense(latent_dim, name='z_mean')(x)
        z_log_var = Dense(latent_dim, name='z_log_var')(x)

        # use reparameterization trick to push the sampling out as input
        # note that "output_shape" isn't necessary with the TensorFlow backend
        z = Lambda(utils_conv2d.sampling, output_shape=(latent_dim,), name='z')([z_mean, z_log_var])

        # instantiate encoder model
        encoder = Model(inputs, [z_mean, z_log_var, z], name='encoder')
        encoder.summary()
        plot_model(encoder, to_file=self.log_dir + 'vae_cnn_encoder.png', show_shapes=True)

        # build decoder model
        latent_inputs = Input(shape=(latent_dim,), name='z_sampling')
        x = Dense(shape[1] * shape[2] * shape[3], activation=self.activation)(latent_inputs)
        x = Reshape((shape[1], shape[2], shape[3]))(x)

        for i in range(2):
            x = Conv2DTranspose(filters=filters,
                                kernel_size=kernel_size,
                                activation=self.activation,
                                strides=2,
                                padding='same')(x)
            filters //= 2

        outputs = Conv2DTranspose(filters=1,
                                kernel_size=kernel_size,
                                activation=self.activation,
                                padding='same',
                                name='decoder_output')(x)

        # instantiate decoder model
        decoder = Model(latent_inputs, outputs, name='decoder')
        decoder.summary()
        plot_model(decoder, to_file=self.log_dir + 'vae_cnn_decoder.png', show_shapes=True)

        # instantiate VAE model
        outputs = decoder(encoder(inputs)[2])
        vae = Model(inputs, outputs, name='vae')
        model = (encoder, decoder)
        reconstruction_loss = mse(K.flatten(inputs), K.flatten(outputs))
        reconstruction_loss *= 160 * 120
        kl_loss = 1 + z_log_var - K.square(z_mean) - K.exp(z_log_var)
        kl_loss = K.sum(kl_loss, axis=-1)
        kl_loss *= -0.5
        vae_loss = K.mean(reconstruction_loss + kl_loss)
        vae.add_loss(vae_loss)
        
        return vae, model
>>>>>>> ac5749f2064cbcefe9a1fb3d671274d69cf2fbe0

    def train(self):
        # self.vae.compile(optimizer=self.optimizer,
        #                    loss=self.loss,
        #                    metrics=['mse', 'mae'])
        
        self.vae.compile(optimizer='rmsprop')

        training_history = self.vae.fit(self.input_train,
                                        #   self.target_train,
                                          batch_size=self.batch_size,
                                          epochs=self.epochs,
                                          callbacks=self.callbacks,
                                          validation_data=(self.input_valid,None))
                                        #   shuffle=True,
                                        #   verbose=2)

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
        self.vae.load_weights(self.log_dir + 'best_model.hdf5')
        self.vae.compile(optimizer='rmsprop')
        # self.vae.compile(optimizer=self.optimizer, loss=self.loss)
        
        input_test = self.input_test
        actual_data = self.target_test
<<<<<<< HEAD
        predicted_data = np.zeros(shape=(len(actual_data), 1, 160,
                                         120, 1))
        from tqdm import tqdm
        iterator = tqdm(range(0, len(actual_data)))
        for i in iterator:
            input = np.zeros(shape=(1, self.seq_len, 160, 120, 1))
            input[0] = input_test[i].copy()
            yhats = self.model.predict(input)
            predicted_data[i, 0] = yhats[0, -1]
=======
        predicted_data = np.zeros(shape=(len(actual_data), 160, 120, 1))
        from tqdm import tqdm
        iterator = tqdm(range(0, len(actual_data)))
        for i in iterator:
            input = np.zeros(shape=(1, 160, 120, 1))
            input[0] = input_test[i].copy()
            yhats = self.vae.predict(input)
            predicted_data[i] = yhats[0]
>>>>>>> ac5749f2064cbcefe9a1fb3d671274d69cf2fbe0

        data_npz = self.config_model['data_kwargs'].get('dataset')
        lon = np.load(data_npz)['input_lon']
        lat = np.load(data_npz)['input_lat']

        gauge_dataset = self.config_model['data_kwargs'].get('gauge_dataset')
        gauge_lon = np.load(gauge_dataset)['gauge_lon']
        gauge_lat = np.load(gauge_dataset)['gauge_lat']
        gauge_precipitation = np.load(gauge_dataset)['gauge_precip']

        gauge_arr = []
        preds_arr = []
<<<<<<< HEAD
        num_gauge = 0
        num_preds = 0
=======
        num_preds = 0
        num_gauge = 0
>>>>>>> ac5749f2064cbcefe9a1fb3d671274d69cf2fbe0
        # MAE for only gauge data
        for i in range(len(gauge_lat)):
            lat = gauge_lat[i]
            lon = gauge_lon[i]
            temp_lat = int(round((23.95 - lat) / 0.1))
            temp_lon = int(round((lon - 100.05) / 0.1))

            # gauge data
            gauge_precip = gauge_precipitation[-354:, i]
            gauge_arr.append(gauge_precip)

            # prediction data
            preds = predicted_data[:, temp_lat, temp_lon, 0]
            preds_arr.append(preds)
            x = np.count_nonzero(preds > 0)
            y = np.count_nonzero(gauge_precip > 0)
<<<<<<< HEAD
            print("Prediction: ", x, "Gauge: ", y)
            num_preds = num_preds + x
            num_gauge = num_gauge + y

        print(num_preds, num_gauge)
=======
            num_preds = num_preds + x            
            num_gauge = num_gauge + y
            print("Prediction: ", x, "Gauge: ", y)

        print(num_preds, num_gauge)        
>>>>>>> ac5749f2064cbcefe9a1fb3d671274d69cf2fbe0
        common_util.cal_error(gauge_arr, preds_arr)

    def plot_result(self):
        from matplotlib import pyplot as plt
        preds = np.load(self.log_dir + 'pd.npy')
        gt = np.load(self.log_dir + 'gt.npy')
        plt.plot(preds[:], label='preds')
        plt.plot(gt[:], label='gt')
        plt.legend()
        plt.savefig(self.log_dir + 'result_predict.png')
        plt.close()
