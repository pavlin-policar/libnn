import numpy as np


def random_normal(shape, scale=0.01):
    return scale * np.random.randn(*shape)


def xavier(shape):
    in_shape, out_shape = shape
    return np.random.randn(in_shape, out_shape) / np.sqrt(in_shape)


def xavier_relu(shape):
    in_shape, out_shape = shape
    return np.random.randn(in_shape, out_shape) / np.sqrt(in_shape / 2)


def ones(shape):
    return np.ones(shape)


def zeros(shape):
    return np.zeros(shape)
