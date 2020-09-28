import argparse
import os
import sys
import tensorflow as tf
import numpy as np
import yaml
import random as rn
from model.conv2d import Conv2DSupervisor


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

    # load config for seq2seq model
    if args.config_file != False:
        with open(args.config_file) as f:
            config = yaml.load(f)

    if args.mode == 'conv2d_train':
        model = Conv2DSupervisor(**config)
        model.train()
    elif args.mode == 'conv2d_test':
        # predict
        model = Conv2DSupervisor(**config)
        model.test_prediction()
        model.plot_result()
    elif args.mode == 'k_fold':
        # predict
        model = Conv2DSupervisor(**config)
        model.cross_validation()
    else:
        raise RuntimeError("Mode needs to be train/evaluate/test!")
