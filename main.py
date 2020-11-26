import argparse
import os
import sys
import tensorflow as tf
import numpy as np
import yaml
import random as rn
from model.conv2d import Conv2DSupervisor
from model.ann import ANNSupervisor
from model.lstm import LSTMSupervisor
from model.ed_lstm import EDLSTMSupervisor


def seed():
    # The below is necessary for starting Numpy generated random numbers
    # in a well-defined initial state.
    np.random.seed(2)
    # The below is necessary for starting core Python generated random numbers
    # in a well-defined state.
    rn.seed(12345)
    # The below tf.set_random_seed() will make random number generation
    # in the TensorFlow backend have a well-defined initial state.
    # For further details, see:
    # https://www.tensorflow.org/api_docs/python/tf/set_random_seed
    tf.random.set_seed(1234)
    # tf.set_random_seed(1234)


if __name__ == '__main__':
    seed()
    sys.path.append(os.getcwd())
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_file',
                        default='config/conv2d_gsmap.yaml',
                        type=str,
                        help='Config file for pretrained model.')
    parser.add_argument('--mode',
                        default='conv2d_train',
                        type=str,
                        help='Run mode.')
    args = parser.parse_args()

    if args.config_file != False:
        with open(args.config_file) as f:
            config = yaml.load(f)

    if args.mode == 'cnn_train':
        model = Conv2DSupervisor(**config)
        model.train()
    elif args.mode == 'cnn_test':
        model = Conv2DSupervisor(**config)
        model.test()
        model.plot_result()
    elif args.mode == 'ann_train':
        model = ANNSupervisor(**config)
        model.train()
    elif args.mode == 'ann_test':
        model = ANNSupervisor(**config)
        model.test()
        model.plot_result()
    elif args.mode == 'lstm_train':
        model = LSTMSupervisor(**config)
        model.train()
    elif args.mode == 'lstm_test':
        model = LSTMSupervisor(**config)
        model.test()
        model.plot_result()
    elif args.mode == 'ed_lstm_train':
        model = EDLSTMSupervisor(True, **config)
        model.train()
    elif args.mode == 'ed_lstm_test':
        model = EDLSTMSupervisor(False, **config)
        model.test_overlap_all()
        model.plot_result()
    else:
        raise RuntimeError("Mode needs to be train/evaluate/test!")
