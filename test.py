import numpy as np
from sklearn.model_selection import KFold
X = np.array([[1, 2], [3, 4], [1, 2], [3, 4], [3, 4], [3, 4]])
y = np.array([0, 0, 1, 1, 0, 1])
skf = KFold(n_splits=2)
skf.get_n_splits(X, y)


for train_index, test_index in skf.split(X):
    print(train_index[0:2])
    X_train = X[train_index[0:2]]