from abc import ABCMeta, abstractmethod

import numpy as np

from libnn.util import ForwardBackwardStorage


class Loss(ForwardBackwardStorage, metaclass=ABCMeta):
    def __call__(self, *args, **kwargs):
        return self.cost(*args, **kwargs)

    @abstractmethod
    def cost(self, y_hat, y):
        pass

    @abstractmethod
    def gradient(self):
        pass


class MeanSquaredError(Loss):
    def cost(self, y_hat, y):
        errors = y_hat - y
        self.save_for_backward(errors)
        return errors.dot(errors) / errors.shape[0]

    def gradient(self):
        errors, = self.saved_tensors
        return errors / errors.shape[0]


class BinaryCrossEntropy(Loss):
    def cost(self, y_hat, y):
        self.save_for_backward(y_hat, y)
        cross_entropy = (y * np.log(y_hat)) + ((1 - y) * np.log(1 - y_hat))
        return -np.mean(cross_entropy)

    def gradient(self):
        y_hat, y = self.saved_tensors
        gradient = y / y_hat - (1 - y) / (1 - y_hat)
        return -gradient / y_hat.shape[0]


class CategoricalCrossEntropy(Loss):
    def cost(self, y_hat, y):
        self.save_for_backward(y_hat, y)
        return -np.mean(np.log(y_hat)[np.arange(y.shape[0]), y.astype(int)])

    def gradient(self):
        y_hat, y = self.saved_tensors
        gradient = np.zeros_like(y_hat)
        mask = np.arange(y.shape[0]), y.astype(int)
        gradient[mask] -= y_hat[mask] ** -1 / y.shape[0]
        return gradient
