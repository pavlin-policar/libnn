import numpy as np

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



class Softmax(Activation):
    def forward(self, X):
        z = np.exp(X - np.max(X, axis=1, keepdims=True))
        return z / np.sum(z, axis=1, keepdims=True)

    def backward(self, X):
        probs = self.forward(X)
        return probs * (1 - probs)

        probs = downstream_gradient
        n_shape = probs.shape[1]

        jacobian = probs[..., :, np.newaxis] * (np.eye(n_shape) - probs[..., np.newaxis, :])

        return jacobian


class ReLU(Module):
    def __init__(self):
        super().__init__()
        self.__X_cache = None

    def forward(self, X):
        self.__X_cache = X
        return np.maximum(X, 0)

    def backward(self, downstream_gradient):
        return np.where(self.__X_cache > 0, downstream_gradient, 0)
