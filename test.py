import numpy as np
from scipy.spatial import cKDTree
x, y = np.mgrid[0:5, 2:8]
tree = cKDTree(np.c_[x.ravel(), y.ravel()])
dd = tree.query([1.1, 3.1], k=3)[1]
print(dd)

ix = np.logical_or([True,False,True], [False,False,False])

print(np.where(np.logical_not(ix))[0])