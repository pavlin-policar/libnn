import numpy as np
from numpy.core.umath_tests import matrix_multiply

from libnn.modules import initializations
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


class PReLU(Module):
    def __init__(self, alpha_init='zeros'):
        super().__init__()

        if callable(alpha_init):
            initialization = alpha_init
        elif hasattr(initializations, alpha_init):
            initialization = getattr(initializations, alpha_init)
        else:
            raise ValueError(
                '`%s` is not a recognized initialization scheme' % alpha_init
            )

        self.alpha = self.trainable(initialization(1))

    def forward(self, X):
        self.save_for_backward(X)
        return np.where(X < 0, self.alpha * X, X)

    def backward(self, downstream_gradient):
        X, = self.saved_tensors

        d_alpha = X.copy()
        d_alpha[d_alpha <= 0] = 0
        d_alpha *= downstream_gradient
        self.alpha.grad = np.sum(d_alpha)

        return np.where(X > 0, downstream_gradient, self.alpha * downstream_gradient)


class ELU(Module):
    def __init__(self, alpha=1.0):
        super().__init__()
        self.alpha = alpha

    def forward(self, X):
        X_mask = X > 0
        result = np.where(X_mask, X, self.alpha * np.exp(X) - self.alpha)
        self.save_for_backward(result, X_mask)
        return result

    def backward(self, downstream_gradient):
        result, X_mask = self.saved_tensors
        return np.where(
            X_mask,
            downstream_gradient,
            (result + self.alpha) * downstream_gradient
        )


class SELU(Module):
    alpha = 1.6732632423543772848170429916717
    scale = 1.0507009873554804934193349852946

    def forward(self, X):
        X_mask = X > 0
        result = np.where(X_mask, X, self.alpha * np.exp(X) - self.alpha)
        self.save_for_backward(X_mask, result)
        return self.scale * result

    def backward(self, downstream_gradient):
        X_mask, result = self.saved_tensors
        return np.where(
            X_mask,
            self.scale * downstream_gradient,
            self.scale * (result + self.alpha) * downstream_gradient
        )
