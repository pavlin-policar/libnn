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
        # Because y contains a one-hot encoded representation of classes and
        # can only contain 0 or 1, we can reduce computation overhead.
        return -np.mean(np.log(y_hat)[y.astype(bool)])

    def gradient(self):
        y_hat, y = self.__y_hat, self.__y
        return - y / y_hat / y.shape[0]
