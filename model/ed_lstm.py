from keras.layers import Dense, LSTM, Input, concatenate, dot, Activation
from keras.models import Sequential, Model
import numpy as np
from model import common_util
import model.utils.ed_lstm as utils_ed_lstm
import os
import yaml
from pandas import read_csv
from keras.utils import plot_model
from keras import backend as K
from model.utils.attention import AttentionLayer

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

        if is_training:
            self.model = self.build_model_prediction(is_training)
        else:
            self.model, self.encoder_model, self.decoder_model = self.build_model_prediction(is_training)

    def _model_construction_test(self, is_training=True):
        # Model
        encoder_inputs = Input(shape=(self.seq_len, self._input_dim), name='encoder_input')
        encoder = LSTM(self.rnn_units, return_sequences=True, return_state=True)
        encoder_outputs, enc_state_h, enc_state_c = encoder(encoder_inputs)
        # encoder_outputs, enc_state_h, enc_state_c = Residual_enc(encoder_inputs, rnn_unit=self.rnn_units,
        #                                                             rnn_depth=self._rnn_layers,
        #                                                             rnn_dropout=self._drop_out)

        encoder_states = [enc_state_h, enc_state_c]

        decoder_inputs = Input(shape=(None, self._output_dim),
                                name='decoder_input')
        decoder_lstm = LSTM(self.rnn_units, return_sequences=True, return_state=True)
        decoder_outputs, decoder_state_h, decoder_state_c = decoder_lstm(decoder_inputs, initial_state=encoder_states)

        # layers_dec, decoder_outputs, dec_state_h, dec_state_c = Residual_dec(decoder_inputs, rnn_unit=self.rnn_units,
        #                                                             rnn_depth=self._rnn_layers,
        #                                                             rnn_dropout=self._drop_out,
        #                                                             init_states=encoder_states)

        # attn_layer = AttentionLayer(input_shape=([self.batch_size, self.seq_len, self.rnn_units],
        #                                             [self.batch_size, self.seq_len, self.rnn_units]),
        #                             name='attention_layer')
        attn_layer = AttentionLayer(name='attention_layer')
        attn_out, attn_states = attn_layer([encoder_outputs, decoder_outputs])
        decoder_outputs = Concatenate(axis=-1, name='concat_layer')([decoder_outputs, attn_out])

        # dense decoder_outputs
        decoder_dense = Dense(self._output_dim, activation='relu')
        decoder_outputs = decoder_dense(decoder_outputs)

        # Define the model that will turn
        # `encoder_input_data` & `decoder_input_data` into `decoder_target_data`
        model = Model([encoder_inputs, decoder_inputs], decoder_outputs)
        if is_training:
            return model
        else:
            self._logger.info("Load model from: {}".format(self._log_dir))
            model.load_weights(self._log_dir + 'best_model.hdf5')
            model.compile(optimizer=self._optimizer, loss='mse', metrics=['mse', 'mae'])
            # --------------------------------------- ENcoder model ----------------------------------------------------
            self.encoder_model = Model(encoder_inputs, [encoder_outputs] + encoder_states)
            plot_model(model=self.encoder_model, to_file=self._log_dir + '/encoder.png', show_shapes=True)

            # --------------------------------------- Decoder model ----------------------------------------------------
            decoder_state_input_h = Input(shape=(self.rnn_units,), name='decoder_state_input_h')
            decoder_state_input_c = Input(shape=(self.rnn_units,), name='decoder_state_input_c')
            decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]

            decoder_outputs, _, _ = layers_dec[0](decoder_inputs, initial_state=decoder_states_inputs)
            for i in range (1, self._rnn_layers):
                d_o, dec_state_h, dec_state_c = layers_dec[i](decoder_outputs)
                decoder_outputs = add([decoder_outputs, d_o])

            decoder_states = [dec_state_h, dec_state_c]

            encoder_inf_states = Input(shape=(self.seq_len, self.rnn_units),
                                       name='encoder_inf_states_input')
            attn_out, attn_states = attn_layer([encoder_inf_states, decoder_outputs])

            decoder_outputs = Concatenate(axis=-1, name='concat')([decoder_outputs, attn_out])
            decoder_dense = Dense(self._output_dim, activation='relu')
            decoder_outputs = decoder_dense(decoder_outputs)
            self.decoder_model = Model(
                [decoder_inputs, encoder_inf_states] + decoder_states_inputs,
                [decoder_outputs] + decoder_states)

            plot_model(model=self.decoder_model, to_file=self._log_dir + '/decoder.png', show_shapes=True)
            return model

    def build_model_prediction(self, is_training):
        encoder_inputs = Input(shape=(self.seq_len, self.input_dim), name='encoder_input')
        encoder = LSTM(self.rnn_units, return_sequences=True, return_state=True)
        encoder_outputs, state_h, state_c = encoder(encoder_inputs)

        encoder_states = [state_h, state_c]

        decoder_inputs = Input(shape=(None, self.output_dim), name='decoder_input')
        decoder_lstm = LSTM(self.rnn_units, return_sequences=True, return_state=True)
        decoder_outputs, decoder_state_h, decoder_state_c = decoder_lstm(decoder_inputs, initial_state=encoder_states)
        # attention = dot([decoder_outputs, encoder_outputs], axes=[2, 2])
        # attention = Activation('softmax')(attention)
        # context = dot([attention, encoder_outputs], axes=[2,1])
        # decoder_outputs = concatenate([context, decoder_outputs])

        # attention
        attn_layer = AttentionLayer(input_shape=([None, self.seq_len, self.rnn_units],
                                                    [None, self.seq_len, self.rnn_units]),
                                    name='attention_layer')
        attn_out, attn_states = attn_layer([encoder_outputs, decoder_outputs])
        decoder_outputs = Concatenate(axis=-1, name='concat_layer')([decoder_outputs, attn_out])
        
        # output
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

            # attention
            encoder_inf_states = Input(shape=(self.seq_len, self.rnn_units),
                                       name='encoder_inf_states_input')
            attn_out, attn_states = attn_layer([encoder_inf_states, decoder_outputs])
            decoder_outputs = Concatenate(axis=-1, name='concat')([decoder_outputs, attn_out])

            # output
            decoder_outputs = decoder_dense(decoder_outputs)

            decoder_model = Model([decoder_inputs] + decoder_states_inputs, [decoder_outputs] + decoder_states)

            plot_model(model=encoder_model, to_file=self.log_dir + '/encoder.png', show_shapes=True)
            plot_model(model=decoder_model, to_file=self.log_dir + '/decoder.png', show_shapes=True)

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

    def test(self):
        # print("Load model from: {}".format(self.log_dir))
        # self.model.load_weights(self.log_dir + 'best_model.hdf5')
        # self.model.compile(optimizer=self.optimizer, loss=self.loss)
        input_encoder_test = self.input_encoder_test
        groundtruth = self.target_decoder_test
        preds = np.zeros(shape=(input_encoder_test.shape[0] + input_encoder_test.shape[1] - 1, 1))
        gt = np.zeros(shape=(input_encoder_test.shape[0] + input_encoder_test.shape[1] - 1, 1))

        from tqdm import tqdm
        iterator = tqdm(range(0, len(groundtruth)))

        for i in tqdm(range(0, len(input_encoder_test), self.horizon)):
            input_model = np.reshape(input_encoder_test[i], (1, input_encoder_test[i].shape[0], input_encoder_test[i].shape[1]))
            yhat = self._predict(input_model)
            preds[i:i+self.horizon] = yhat
            gt[i:i+self.horizon] = groundtruth[i]
        
        print(preds)
        
        scaler = self.data["scaler"]
        col = 1
        correct_shape_gt = np.empty(shape=(int(preds.shape[0]/col), col))
        correct_shape_pd = np.empty(shape=(int(preds.shape[0]/col), col))
        for i in range(int(gt.shape[0]/col)):
            correct_shape_gt[i, :] = np.transpose(gt[i*col:(i+1)*col])
            correct_shape_pd[i, :] = np.transpose(preds[i*col:(i+1)*col])

        print(correct_shape_gt.shape)
        print(correct_shape_pd.shape)
        reverse_groundtruth = scaler.inverse_transform(correct_shape_gt)
        reverse_preds = scaler.inverse_transform(correct_shape_pd)
        list_metrics = np.zeros(shape=(1, 2))
        list_metrics[0, 0] = common_util.mae(reverse_groundtruth, reverse_preds)
        list_metrics[0, 1] = common_util.rmse(reverse_groundtruth, reverse_preds)

        np.savetxt(self.log_dir + 'groundtruth.csv', reverse_groundtruth, delimiter=",")
        np.savetxt(self.log_dir + 'preds.csv', reverse_preds, delimiter=",")
        np.savetxt(self.log_dir + 'list_metrics.csv', list_metrics, delimiter=",")

    def test_overlap(self):
        # print("Load model from: {}".format(self.log_dir))
        # self.model.load_weights(self.log_dir + 'best_model.hdf5')
        # self.model.compile(optimizer=self.optimizer, loss=self.loss)
        input_encoder_test = self.input_encoder_test
        groundtruth = self.target_decoder_test
        preds = np.zeros(shape=(input_encoder_test.shape[0] + input_encoder_test.shape[1] - 1, 1))
        gt = np.zeros(shape=(input_encoder_test.shape[0] + input_encoder_test.shape[1] - 1, 1))

        from tqdm import tqdm
        iterator = tqdm(range(0, len(groundtruth)))

        for i in tqdm(range(0, len(input_encoder_test), self.horizon)):
            input_model = np.reshape(input_encoder_test[i], (1, input_encoder_test[i].shape[0], input_encoder_test[i].shape[1]))
            yhat = self._predict(input_model)
            preds[i:i+self.horizon] = yhat[-1]
            gt[i:i+self.horizon] = groundtruth[i, -1]
        
        scaler = self.data["scaler"]
        col = 1
        correct_shape_gt = np.empty(shape=(int(preds.shape[0]/col), col))
        correct_shape_pd = np.empty(shape=(int(preds.shape[0]/col), col))
        for i in range(int(gt.shape[0]/col)):
            correct_shape_gt[i, :] = np.transpose(gt[i*col:(i+1)*col])
            correct_shape_pd[i, :] = np.transpose(preds[i*col:(i+1)*col])

        print(correct_shape_gt.shape)
        print(correct_shape_pd.shape)
        reverse_groundtruth = scaler.inverse_transform(np.tile(correct_shape_gt, (1,72)))
        reverse_preds = scaler.inverse_transform(np.tile(correct_shape_pd, (1,72)))
        list_metrics = np.zeros(shape=(1, 2))
        list_metrics[0, 0] = common_util.mae(reverse_groundtruth[:, 0], reverse_preds[:, 0])
        list_metrics[0, 1] = common_util.rmse(reverse_groundtruth[:, 0], reverse_preds[:, 0])

        np.savetxt(self.log_dir + 'groundtruth.csv', reverse_groundtruth, delimiter=",")
        np.savetxt(self.log_dir + 'preds.csv', reverse_preds, delimiter=",")
        np.savetxt(self.log_dir + 'list_metrics.csv', list_metrics, delimiter=",")
    
    def test_overlap_all(self):
        # print("Load model from: {}".format(self.log_dir))
        # self.model.load_weights(self.log_dir + 'best_model.hdf5')
        # self.model.compile(optimizer=self.optimizer, loss=self.loss)
        input_encoder_test = self.input_encoder_test
        groundtruth = self.target_decoder_test
        preds = np.zeros(shape=(input_encoder_test.shape[0] + input_encoder_test.shape[1] - 1, 1))
        gt = np.zeros(shape=(input_encoder_test.shape[0] + input_encoder_test.shape[1] - 1, 1))

        from tqdm import tqdm
        iterator = tqdm(range(0, len(groundtruth)))

        for i in tqdm(range(0, len(input_encoder_test), self.horizon)):
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

        print(correct_shape_gt.shape)
        print(correct_shape_pd.shape)
        reverse_groundtruth = scaler.inverse_transform(correct_shape_gt)
        reverse_preds = scaler.inverse_transform(correct_shape_pd)
        list_metrics = np.zeros(shape=(1, 2))
        list_metrics[0, 0] = common_util.mae(reverse_groundtruth, reverse_preds)
        list_metrics[0, 1] = common_util.rmse(reverse_groundtruth, reverse_preds)

        np.savetxt(self.log_dir + 'groundtruth.csv', reverse_groundtruth, delimiter=",")
        np.savetxt(self.log_dir + 'preds.csv', reverse_preds, delimiter=",")
        np.savetxt(self.log_dir + 'list_metrics.csv', list_metrics, delimiter=",")
    
    def test_all(self):

        input_encoder_test = self.input_encoder_test
        groundtruth = self.target_decoder_test
        preds = np.zeros(shape=(input_encoder_test.shape[0] + input_encoder_test.shape[1] - 1, input_encoder_test.shape[2]))
        gt = np.zeros(shape=(input_encoder_test.shape[0] + input_encoder_test.shape[1] - 1, input_encoder_test.shape[2]))

        from tqdm import tqdm
        iterator = tqdm(range(0, len(groundtruth)))

        for i in tqdm(range(0, len(input_encoder_test), self.horizon)):
            input_model = np.reshape(input_encoder_test[i], (1, input_encoder_test[i].shape[0], input_encoder_test[i].shape[1]))
            yhat = self._predict(input_model)
            preds[i:i+self.horizon] = yhat[-1]
            gt[i:i+self.horizon] = groundtruth[i, -1]
        
        scaler = self.data["scaler"]
        # for i in range(int(gt.shape[0]/col)):
        #     correct_shape_gt[i, :] = np.transpose(gt[i*col:(i+1)*col])
        #     correct_shape_pd[i, :] = np.transpose(preds[i*col:(i+1)*col])

        # print(correct_shape_gt.shape)
        # print(correct_shape_pd.shape)
        reverse_groundtruth = scaler.inverse_transform(gt)
        reverse_preds = scaler.inverse_transform(preds)
        list_metrics = np.zeros(shape=(1, 2))
        list_metrics[0, 0] = common_util.mae(reverse_groundtruth, reverse_preds)
        list_metrics[0, 1] = common_util.rmse(reverse_groundtruth, reverse_preds)

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

        plt.plot(preds[-1000:, 2], label='preds')
        plt.plot(gt[-1000:, 2], label='gt')
        plt.legend()
        plt.savefig(self.log_dir + 'result_predict.png')
        plt.close()

    def cross_validation(self, **kwargs):
        from sklearn.model_selection import KFold
        kfold = KFold(n_splits=5, shuffle=True, random_state=2)
        input_data, target_data = utils_ed_lstm.create_data_prediction(**kwargs)
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

            with open("config/lstm.yaml") as f:
                config = yaml.load(f)    
            config['base_dir'] = "log/lstm/" + str(count) + '/'

            self.config_model = common_util.get_config_model(**config)
            self.log_dir = self.config_model['log_dir']
            self.callbacks = self.config_model['callbacks']
            self.model = self.build_model_prediction()
            self.train()
            self.test()
            print("Complete " + str(count) + " !!!!")