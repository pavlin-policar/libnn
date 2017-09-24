import numpy as np


def xor_data(n_samples):
    X = np.random.randn(n_samples, 2)
    y = np.logical_xor(X[:, 0] > 0, X[:, 1] > 0).astype(np.uint8)

    return X, y


def spirals(n_samples, dimensionality=2, n_classes=3, spread=0.2):
    X = np.zeros((n_samples * n_classes, dimensionality))
    y = np.zeros(n_samples * n_classes, dtype=np.uint8)

    for j in range(n_classes):
        ix = range(n_samples * j, n_samples * (j + 1))
        r = np.linspace(0, 1, n_samples)
        t = np.linspace(j * 4, (j + 1) * 4, n_samples)
        t += np.random.randn(n_samples) * spread

        X[ix] = np.c_[r * np.sin(t), r * np.cos(t)]
        y[ix] = j

    return X, y
