import numpy as np
from numpy.core.umath_tests import matrix_multiply

from libnn.modules.module import Module


class Sigmoid(Module):
    def __init__(self):
        super().__init__()
        self.__z = None

    def forward(self, X):
        result = 1 / (1 + np.exp(-X))
        self.__z = result
        return result

    def backward(self, downstream_gradient):
        z = self.__z
        return downstream_gradient * z * (1 - z)


class Softmax(Module):
    def __init__(self):
        super().__init__()
        self.__probabilities = None

    def forward(self, X):
        z = np.exp(X - np.max(X, axis=1, keepdims=True))
        probabilities = z / np.sum(z, axis=1, keepdims=True)
        if self.training:
            self.__probabilities = probabilities
        return probabilities

    def backward(self, downstream_gradient):
        probs = self.__probabilities

        n_shape = probs.shape[1]

        jacobian = probs[..., :, np.newaxis] * (np.eye(n_shape) - probs[..., np.newaxis, :])

        # Downstream gradient is 2d, jacobian is 3d, and we need to perform
        # matrix-vector multiplication jacobian[i] * dL[i]. Since the jacobian
        # is symmetric, we can omit the transpose of the jacobian
        product = matrix_multiply(jacobian, downstream_gradient[..., np.newaxis])
        # matrix_multiply returns a 3d tensor, however we need a 2d matrix
        product = product.squeeze()

        return product


class ReLU(Module):
    def __init__(self):
        super().__init__()
        self.__X_cache = None

    def forward(self, X):
        self.__X_cache = X
        return np.maximum(X, 0)

    def backward(self, downstream_gradient):
        return np.where(self.__X_cache > 0, downstream_gradient, 0)
