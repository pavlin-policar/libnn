import numpy as np
from numpy.core.umath_tests import matrix_multiply

from libnn.modules.module import Module


class Sigmoid(Module):
    def forward(self, X):
        result = 1 / (1 + np.exp(-X))
        self.save_for_backward(result)
        return result

    def backward(self, downstream_gradient):
        z, = self.saved_tensors
        return downstream_gradient * z * (1 - z)


class Tanh(Module):
    def forward(self, X):
        result = np.tanh(X)
        self.save_for_backward(result)
        return result

    def backward(self, downstream_gradient):
        result, = self.saved_tensors
        local_gradient = 1 - result ** 2
        return downstream_gradient * local_gradient


class Softmax(Module):
    def forward(self, X):
        z = np.exp(X - np.max(X, axis=1, keepdims=True))
        probabilities = z / np.sum(z, axis=1, keepdims=True)
        self.save_for_backward(probabilities)
        return probabilities

    def backward(self, downstream_gradient):
        probs, = self.saved_tensors
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
    def forward(self, X):
        self.save_for_backward(X)
        return np.maximum(X, 0)

    def backward(self, downstream_gradient):
        X, = self.saved_tensors
        return np.where(X > 0, downstream_gradient, 0)


class LeakyReLU(Module):
    def __init__(self, alpha=0.01):
        super().__init__()
        self.alpha = alpha

    def forward(self, X):
        self.save_for_backward(X)
        return np.where(X < 0, self.alpha * X, X)

    def backward(self, downstream_gradient):
        X, = self.saved_tensors
        return np.where(
            X > 0,
            downstream_gradient,
            self.alpha * downstream_gradient
        )

