import numpy as np


def random_normal(in_shape, out_shape, scale=0.01):
    return scale * np.random.randn(in_shape, out_shape)


def xavier(in_shape, out_shape):
    return np.random.randn(in_shape, out_shape) / np.sqrt(in_shape)


def xavier_relu(in_shape, out_shape):
    return np.random.randn(in_shape, out_shape) / np.sqrt(in_shape / 2)


def ones(in_shape, out_shape):
    return np.ones((in_shape, out_shape))


def zeros(in_shape, out_shape):
    return np.zeros((in_shape, out_shape))
