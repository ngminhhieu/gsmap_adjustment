from tensorflow.keras.layers import Dense, LSTM, Input, Concatenate, dot, Activation
from tensorflow.keras.models import Sequential, Model
import numpy as np
from model import common_util
import model.utils.ed_lstm as utils_ed_lstm
import os
import yaml
from pandas import read_csv
from tensorflow.keras.utils import plot_model
from tensorflow.keras import backend as K
from model.utils.attention import AttentionLayer
from tqdm import tqdm
import datetime


class EDLSTMSupervisor():
    def __init__(self, is_training=True, **kwargs):
        self.config_model = common_util.get_config_model(**kwargs)

        # load_data
        self.data = utils_ed_lstm.load_dataset(**kwargs)
        self.input_encoder_train = self.data['input_encoder_train']
        self.input_encoder_valid = self.data['input_encoder_valid']
        self.input_encoder_test = self.data['input_encoder_test']
        self.input_decoder_train = self.data['input_decoder_train']
        self.input_decoder_valid = self.data['input_decoder_valid']
        self.input_decoder_test = self.data['input_decoder_test']
        self.target_decoder_train = self.data['target_decoder_train']
        self.target_decoder_valid = self.data['target_decoder_valid']
        self.target_decoder_test = self.data['target_decoder_test']

        # other configs
        self.rnn_units = self.config_model['rnn_units']
        self.log_dir = self.config_model['log_dir']
        self.optimizer = self.config_model['optimizer']
        self.loss = self.config_model['loss']
        self.activation = self.config_model['activation']
        self.batch_size = self.config_model['batch_size']
        self.epochs = self.config_model['epochs']
        self.callbacks = self.config_model['callbacks']
        self.seq_len = self.config_model['seq_len']
        self.horizon = self.config_model['horizon']
        self.input_dim = self.config_model['input_dim']
        self.output_dim = self.config_model['output_dim']
        self.dropout = self.config_model['dropout']

        if is_training:
            self.model = self.build_model_prediction(is_training)
        else:
            self.model, self.encoder_model, self.decoder_model = self.build_model_prediction(is_training)

    def build_model_prediction(self, is_training):
        encoder_inputs = Input(shape=(self.seq_len, self.input_dim), name='encoder_input')
        encoder = LSTM(self.rnn_units, return_state=True, dropout=self.dropout)
        encoder_outputs, state_h, state_c = encoder(encoder_inputs)

        encoder_states = [state_h, state_c]

        decoder_inputs = Input(shape=(None, self.output_dim), name='decoder_input')
        decoder_lstm = LSTM(self.rnn_units, return_sequences=True, return_state=True, dropout=self.dropout)
        decoder_outputs, decoder_state_h, decoder_state_c = decoder_lstm(decoder_inputs, initial_state=encoder_states)

        decoder_dense = Dense(self.output_dim, activation='relu')
        decoder_outputs = decoder_dense(decoder_outputs)
        model = Model([encoder_inputs, decoder_inputs], decoder_outputs)

        if is_training:
            return model
        else:
            print("Load model from: {}".format(self.log_dir))
            model.load_weights(self.log_dir + 'best_model.hdf5')
            model.compile(optimizer=self.optimizer, loss='mse')

            # Inference encoder_model
            encoder_model = Model(encoder_inputs, encoder_states)

            # Inference decoder_model
            decoder_state_input_h = Input(shape=(self.rnn_units,))
            decoder_state_input_c = Input(shape=(self.rnn_units,))
            decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]
            decoder_outputs, state_h, state_c = decoder_lstm(decoder_inputs, initial_state=decoder_states_inputs)
            decoder_states = [state_h, state_c]
            decoder_outputs = decoder_dense(decoder_outputs)

            decoder_model = Model([decoder_inputs] + decoder_states_inputs, [decoder_outputs] + decoder_states)

            # plot_model(model=encoder_model, to_file=self.self.log_dir + '/encoder.png', show_shapes=True)
            # plot_model(model=decoder_model, to_file=self.log_dir + '/decoder.png', show_shapes=True)

            return model, encoder_model, decoder_model

    def train(self):
        self.model.compile(optimizer=self.optimizer,
                           loss=self.loss,
                           metrics=['mse', 'mae'])

        training_history = self.model.fit([self.input_encoder_train, self.input_decoder_train],
                                          self.target_decoder_train,
                                          batch_size=self.batch_size,
                                          epochs=self.epochs,
                                          callbacks=self.callbacks,
                                          validation_data=([self.input_encoder_valid, self.input_decoder_valid],
                                                           self.target_decoder_valid),
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
    
    def test_overlap_all(self):
        input_encoder_test = self.input_encoder_test
        input_encoder_test = input_encoder_test[:200]
        groundtruth = self.target_decoder_test
        preds = np.zeros(shape=(input_encoder_test.shape[0] + input_encoder_test.shape[1] - 1, 1))
        gt = np.zeros(shape=(input_encoder_test.shape[0] + input_encoder_test.shape[1] - 1, 1))

        iterator = tqdm(range(0, len(input_encoder_test), self.horizon))
        for i in iterator:
            input_model = np.reshape(input_encoder_test[i], (1, input_encoder_test[i].shape[0], input_encoder_test[i].shape[1]))
            yhat = self._predict(input_model)
            preds[i:i+self.horizon] = yhat[-1]
            gt[i:i+self.horizon] = groundtruth[i, -1]
            
        scaler = self.data["scaler"]
        col = 72
        correct_shape_gt = np.empty(shape=(int(preds.shape[0]/col), col))
        correct_shape_pd = np.empty(shape=(int(preds.shape[0]/col), col))
        for i in range(int(gt.shape[0]/col)):
            correct_shape_gt[i, :] = np.transpose(gt[i*col:(i+1)*col])
            correct_shape_pd[i, :] = np.transpose(preds[i*col:(i+1)*col])

        reverse_groundtruth = scaler.inverse_transform(correct_shape_gt)
        reverse_preds = scaler.inverse_transform(correct_shape_pd)
        list_metrics = np.zeros(shape=(1, 3))
        list_metrics[0, 0] = common_util.mae(reverse_groundtruth, reverse_preds)
        list_metrics[0, 1] = common_util.rmse(reverse_groundtruth, reverse_preds)
        list_metrics[0, 2] = common_util.nashsutcliffe(reverse_groundtruth, reverse_preds)
        list_metrics = list_metrics.tolist()
        common_util.save_metrics(self.log_dir + "list_metrics.csv", list_metrics)

        np.savetxt(self.log_dir + 'groundtruth.csv', reverse_groundtruth, delimiter=",")
        np.savetxt(self.log_dir + 'preds.csv', reverse_preds, delimiter=",")
        # np.savetxt(self.log_dir + 'list_metrics.csv', list_metrics, delimiter=",")
    
    def test_all(self):

        input_encoder_test = self.input_encoder_test
        groundtruth = self.target_decoder_test
        preds = np.zeros(shape=(input_encoder_test.shape[0] + input_encoder_test.shape[1] - 1, input_encoder_test.shape[2]))
        gt = np.zeros(shape=(input_encoder_test.shape[0] + input_encoder_test.shape[1] - 1, input_encoder_test.shape[2]))

        for i in tqdm(range(0, len(input_encoder_test), self.horizon)):
            input_model = np.reshape(input_encoder_test[i], (1, input_encoder_test[i].shape[0], input_encoder_test[i].shape[1]))
            yhat = self._predict(input_model)
            preds[i:i+self.horizon] = yhat[-1]
            gt[i:i+self.horizon] = groundtruth[i, -1]
        
        scaler = self.data["scaler"]
        reverse_groundtruth = scaler.inverse_transform(gt)
        reverse_preds = scaler.inverse_transform(preds)
        list_metrics = np.zeros(shape=(1, 3))
        list_metrics[0, 0] = common_util.mae(reverse_groundtruth, reverse_preds)
        list_metrics[0, 1] = common_util.rmse(reverse_groundtruth, reverse_preds)
        list_metrics[0, 2] = common_util.nashsutcliffe(reverse_groundtruth, reverse_preds)
        list_metrics = list_metrics.tolist()
        list_metrics = [str(datetime.datetime.now())] + list_metrics

        np.savetxt(self.log_dir + 'groundtruth.csv', reverse_groundtruth, delimiter=",")
        np.savetxt(self.log_dir + 'preds.csv', reverse_preds, delimiter=",")
        np.savetxt(self.log_dir + 'list_metrics.csv', list_metrics, delimiter=",")


    def _predict(self, source):
        states_value = self.encoder_model.predict(source)
        target_seq = np.zeros((1, 1, self.output_dim))
        preds = np.zeros(shape=(self.horizon, self.output_dim),
                        dtype='float32')
        for i in range(self.horizon):
            output = self.decoder_model.predict([target_seq] + states_value)
            yhat = output[0]
            # store prediction
            preds[i] = yhat
            # update target sequence
            target_seq = yhat
            # Update states
            states_value = output[1:]
        return preds


    def plot_result(self):
        from matplotlib import pyplot as plt
        preds = read_csv(self.log_dir + 'preds.csv')
        gt = read_csv(self.log_dir + 'groundtruth.csv')
        preds = preds.to_numpy()
        gt = gt.to_numpy()

        for i in range(preds.shape[1]):
            plt.plot(preds[-1000:, i], label='preds')
            plt.plot(gt[-1000:, i], label='gt')
            plt.legend()
            plt.savefig(self.log_dir + 'result_predict_{}.png'.format(str(i)))
            plt.close()