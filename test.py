import numpy as np
from matplotlib import pyplot as plt
from numpy.random import rand

a = np.array([[1,2,4], [4,5,6], [10,20,30]])
b = a[0:2, :]*3
print(b)

z = np.array([[[1,3,4,50]]], [[[0.1,0,0.2,0.3]]])
print(z.shape)
z = np.reshape(z, (4,2))
print(z)