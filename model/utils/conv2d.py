import numpy as np
from model import common_util
from sklearn.preprocessing import MinMaxScaler
from keras import backend as K

def create_data_prediction(**kwargs):

    data_npz = kwargs['data'].get('dataset')
    seq_len = kwargs['model'].get('seq_len')
    horizon = kwargs['model'].get('horizon')

    time = np.load(data_npz)['time']
    # horizon is in seq_len. the last
    T = len(time)

    lon = np.load(data_npz)['output_lon']
    lat = np.load(data_npz)['output_lat']
    precip = np.load(data_npz)['output_precip']

    input_conv2d_gsmap = np.zeros(shape=(T, len(lat), len(lon), 1))
    target_conv2d_gsmap = np.zeros(shape=(T, len(lat), len(lon), 1))

    dataset = '../../data/conv2d_gsmap/npz/gauge_data.npz'
    gauge_lon = np.load(gauge_dataset)['gauge_lon']
    gauge_lat = np.load(gauge_dataset)['gauge_lat']
    gauge_precipitation = np.load(gauge_dataset)['gauge_precip']
    for i in range(len(gauge_lat)):
        lat = gauge_lat[i]
        lon = gauge_lon[i]
        temp_lat = int(round((23.95 - lat) / 0.1))
        temp_lon = int(round((lon - 100.05) / 0.1))
        gauge_precip = gauge_precipitation[:, i]
        gsmap_precip = precip[:, temp_lat, temp_lon]
        input_conv2d_gsmap[:, temp_lat, temp_lon, 0] = gsmap_precip
        target_conv2d_gsmap[:, temp_lat, temp_lon, 0] = gauge_precip
    return input_conv2d_gsmap, target_conv2d_gsmap


def load_dataset(**kwargs):
    # get preprocessed input and target
    input_conv2d_gsmap, target_conv2d_gsmap = create_data_prediction(**kwargs)

    # get test_size, valid_size from config
    test_size = kwargs['data'].get('test_size')
    valid_size = kwargs['data'].get('valid_size')

    # split data to train_set, valid_set, test_size
    input_train, input_valid, input_test = common_util.prepare_train_valid_test(
        input_conv2d_gsmap, test_size=test_size, valid_size=valid_size)
    target_train, target_valid, target_test = common_util.prepare_train_valid_test(
        target_conv2d_gsmap, test_size=test_size, valid_size=valid_size)
    data = {}
    for cat in ["train", "valid", "test"]:
        x, y = locals()["input_" + cat], locals()["target_" + cat]
        data["input_" + cat] = x
        data["target_" + cat] = y

    return data

# reparameterization trick
# instead of sampling from Q(z|X), sample eps = N(0,I)
# then z = z_mean + sqrt(var)*eps
def sampling(args):
    """Reparameterization trick by sampling fr an isotropic unit Gaussian.

    # Arguments
        args (tensor): mean and log of variance of Q(z|X)

    # Returns
        z (tensor): sampled latent vector
    """

    z_mean, z_log_var = args
    batch = K.shape(z_mean)[0]
    dim = K.int_shape(z_mean)[1]
    # by default, random_normal has mean=0 and std=1.0
    epsilon = K.random_normal(shape=(batch, dim))
    return z_mean + K.exp(0.5 * z_log_var) * epsilon