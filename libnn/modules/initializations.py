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


def he_normal(shape):
    """Initialization scheme for rectified linear units.

    Parameters
    ----------
    shape : Tuple

    Returns
    -------
    np.ndarray

    References
    ----------
    He et al., http://arxiv.org/abs/1502.01852

    """
    in_shape, out_shape = shape
    return np.random.normal(0., np.sqrt(2 / in_shape), shape)
