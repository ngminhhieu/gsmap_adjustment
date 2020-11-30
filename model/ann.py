from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential
import numpy as np
from model import common_util
import model.utils.ann as utils_ann
import os
import yaml
from pandas import read_csv
from tensorflow.keras.utils import plot_model
from tensorflow.keras import backend as K
from tqdm import tqdm


class ANNSupervisor():
    def __init__(self, **kwargs):
        self.config_model = common_util.get_config_model(**kwargs)

        # load_data
        self.data = utils_ann.load_dataset(**kwargs)
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
        model = Sequential()
        model.add(Dense(20, input_dim=1, activation=self.activation))
        model.add(Dense(10, activation=self.activation))
        model.add(Dense(1, activation=self.activation))
        print(model.summary())

        # plot model
        plot_model(model=model,
                   to_file=self.log_dir + '/model.png',
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
                                          verbose=0)

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
        actual_data = self.target_test
        predicted_data = np.zeros(shape=(len(actual_data), 1))
        iterator = tqdm(range(0, len(actual_data)))
        for i in iterator:
            input = np.zeros(shape=(1, 1))
            input = input_test[i].copy()
            yhats = self.model.predict(input)
            predicted_data[i] = yhats[0]

        scaler = self.data["scaler"]
        reverse_actual_data = scaler.inverse_transform(actual_data)
        reverse_predicted_data = scaler.inverse_transform(predicted_data)
        list_metrics = np.zeros(shape=(1, 3))
        list_metrics[0, 0] = common_util.mae(reverse_actual_data, reverse_predicted_data)
        list_metrics[0, 1] = common_util.rmse(reverse_actual_data, reverse_predicted_data)
        list_metrics[0, 2] = common_util.nashsutcliffe(reverse_actual_data, reverse_predicted_data)
        list_metrics = list_metrics.tolist()
        common_util.save_metrics(self.log_dir + "list_metrics.csv", list_metrics)
        np.savetxt(self.log_dir + 'groundtruth.csv', reverse_groundtruth, delimiter=",")
        np.savetxt(self.log_dir + 'preds.csv', reverse_preds, delimiter=",")

    def plot_result(self):
        from matplotlib import pyplot as plt
        preds = read_csv(self.log_dir + 'preds.csv')
        gt = read_csv(self.log_dir + 'groundtruth.csv')
        preds = preds.to_numpy()
        gt = gt.to_numpy()
        plt.plot(preds[:], label='preds')
        plt.plot(gt[:], label='gt')
        plt.legend()
        plt.savefig(self.log_dir + 'result_predict.png')
        plt.close()

    def cross_validation(self, **kwargs):
        from sklearn.model_selection import KFold
        kfold = KFold(n_splits=5, shuffle=True, random_state=2)
        input_data, target_data = utils_ann.create_data_prediction(**kwargs)
        count = 0
        for train_index, test_index in kfold.split(input_data):
            count += 1
            pivot = int(0.8*len(train_index))
            input_train = input_data[train_index[0:pivot]]
            input_valid = input_data[train_index[pivot:]]
            input_test = input_data[test_index]

            target_train = target_data[train_index[0:pivot]]
            target_valid = target_data[train_index[pivot:]]
            target_test = target_data[test_index]

            self.input_train = input_train
            self.input_valid = input_valid
            self.input_test = input_test
            self.target_train = target_train
            self.target_valid = target_valid
            self.target_test = target_test

            with open("config/ann.yaml") as f:
                config = yaml.load(f)    
            config['base_dir'] = "log/ann/" + str(count) + '/'

            self.config_model = common_util.get_config_model(**config)
            self.log_dir = self.config_model['log_dir']
            self.callbacks = self.config_model['callbacks']
            self.model = self.build_model_prediction()
            self.train()
            self.test()
            print("Complete " + str(count) + " !!!!")