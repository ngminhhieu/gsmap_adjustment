from keras.models import Sequential
from keras.layers import Flatten, Dense, MaxPooling2D, Conv2D, Conv2DTranspose, UpSampling2D, Cropping2D
import numpy as np
import pylab as plt
from model import common_util
import model.utils.conv2d as utils_conv2d
import os
import yaml


class Conv2DSupervisor():
    def __init__(self, is_training=True, **kwargs):
        self.config_model = common_util.get_config_model(**kwargs)

        # write function load_data here
        self.data = utils_conv2d.load_dataset(**kwargs)
        self.model = self.build_model()

    def build_model(self):
        model = Sequential()

        # extract useful information
        model.add(
            Conv2D(filters=16,
                   kernel_size=(3, 3),
                   padding='same',
                   activation=self.config_model['activation'],
                   input_shape=(72, 72, 3)))
        model.add(MaxPooling2D(pool_size=(2, 2)))

        model.add(
            Conv2D(
                filters=32,
                kernel_size=(3, 3),
                padding='same',
                activation=self.config_model['activation'],
            ))
        model.add(MaxPooling2D(pool_size=(2, 2)))

        # model.add(
        #     Conv2D(filters=64,
        #            kernel_size=(3, 3),
        #            padding='same',
        #            activation=self.config_model['activation']))
        # model.add(MaxPooling2D(pool_size=(2, 2)))

        # scaling up
        model.add(
            Conv2DTranspose(
                filters=32,
                kernel_size=(3, 3),
                strides=(10, 7),
                #  padding='same',
                activation=self.config_model['activation']))

        model.add(
            Conv2D(filters=32,
                   kernel_size=(3, 3),
                   padding='same',
                   activation=self.config_model['activation']))

        # ((top_crop, bottom_crop), (left_crop, right_crop))
        model.add(Cropping2D(cropping=((10, 10), (3, 3))))

        model.add(
            Conv2D(filters=3,
                   kernel_size=(3, 3),
                   padding='same',
                   activation=self.config_model['activation']))
        # model.add(MaxPooling2D(pool_size=(2, 2)))

        # model.add(
        #     Conv2DTranspose(filters=64,
        #                     kernel_size=(3, 3),
        #                     padding='same',
        #                     strides=(5, 4),
        #                     activation=self.config_model['activation']))
        # model.add(UpSampling2D(size=(2, 2)))
        # model.add(
        #     Conv2DTranspose(filters=32,
        #                     kernel_size=(3, 3),
        #                     padding='same',
        #                     activation=self.config_model['activation']))
        # model.add(UpSampling2D(size=(2, 2)))
        # model.add(
        #     Conv2DTranspose(filters=3,
        #                     kernel_size=(3, 3),
        #                     padding='same',
        #                     activation=self.config_model['activation']))
        print(model.summary())

        return model

    def train(self):
        self.model.compile(optimizer=self.config_model['optimizer'],
                           loss=self.config_model['loss'],
                           metrics=['mse', 'mae'])

        training_history = self.model.fit(
            self.data['input_train'],
            self.data['target_train'],
            batch_size=self.config_model['batch_size'],
            epochs=self.config_model['epochs'],
            callbacks=self.config_model['callbacks'],
            validation_data=(self.data['input_valid'],
                             self.data['target_valid']),
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
            config['train']['log_dir'] = self.config_model['log_dir']
            with open(
                    os.path.join(self.config_model['log_dir'],
                                 config_filename), 'w') as f:
                yaml.dump(config, f, default_flow_style=False)

    def test(self):
        pass
