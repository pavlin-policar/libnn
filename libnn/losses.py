from abc import ABCMeta, abstractmethod

import numpy as np


class Loss(metaclass=ABCMeta):
    def __call__(self, *args, **kwargs):
        return self.cost(*args, **kwargs)

    @abstractmethod
    def cost(self, y_hat, y):
        pass

    @abstractmethod
    def gradient(self):
        pass


class MeanSquaredError(Loss):
    def __init__(self):
        self.__errors = None

    def cost(self, y_hat, y):
        errors = y_hat - y
        self.__errors = errors
        return errors.dot(errors) / errors.shape[0]

    def gradient(self):
        return self.__errors / self.__errors.shape[0]


class BinaryCrossEntropy(Loss):
    def __init__(self):
        self.__y_hat = self.__y = None

    def cost(self, y_hat, y):
        self.__y_hat, self.__y = y_hat, y
        cross_entropy = (y * np.log(y_hat)) + ((1 - y) * np.log(1 - y_hat))
        return -np.mean(cross_entropy)

    def gradient(self):
        y_hat, y = self.__y_hat, self.__y
        gradient = y / y_hat - (1 - y) / (1 - y_hat)
        return -gradient / y_hat.shape[0]


class CategoricalCrossEntropy(Loss):
    def __init__(self):
        self.__y_hat = self.__y = None

    def cost(self, y_hat, y):
        self.__y_hat, self.__y = y_hat, y
        return -np.mean(np.log(y_hat)[np.arange(y.shape[0]), y.astype(int)])

    def gradient(self):
        y_hat, y = self.__y_hat, self.__y
        gradient = np.zeros_like(y_hat)
        mask = np.arange(y.shape[0]), y.astype(int)
        gradient[mask] -= y_hat[mask] ** -1 / y.shape[0]
        return gradient
