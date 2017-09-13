import numpy as np


def xor_data(n_samples):
    X = np.random.randn(n_samples, 2)
    y = np.logical_xor(X[:, 0] > 0, X[:, 1] > 0).astype(np.uint8)

    return X, y
