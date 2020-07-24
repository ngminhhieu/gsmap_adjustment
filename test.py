import numpy as np
from matplotlib import pyplot as plt
from numpy.random import rand

data = np.load('data/conv2d_gsmap/map_gauge_72_stations_maemin_r05.npz')
data_2 = np.load('data/conv2d_gsmap/map_gauge_72_stations_maemin_r05.npz')
print(data['map_lon'])
print(data['map_lat'])

print(data['gauge_lon'])
print(data['gauge_lat'])