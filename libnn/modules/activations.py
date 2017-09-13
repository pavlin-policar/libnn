import numpy as np

from libnn.modules.module import Module


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
        num_samples = self.__probabilities.shape[0]

        grad = self.__probabilities.copy()
        grad[range(num_samples), downstream_gradient] -= 1
        grad /= num_samples

        return grad
