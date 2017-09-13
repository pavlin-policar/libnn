import numpy as np


class TrainableParameter(np.ndarray):
    def __new__(cls, data, **kwargs):
        self = super().__new__(cls, data.shape, **kwargs)

        self[:] = data
        self.grad = None

        return self


class Module:
    def __init__(self):
        self.training = True
        self.trainable_parameters = []

    def trainable(self, *args, **kwargs):
        param = TrainableParameter(*args, **kwargs)
        self.trainable_parameters.append(param)
        return param

    def forward(self, X, y=None):
        ...

    def backward(self, downstream_gradient):
        ...

    @property
    def train(self):
        self.training = True

    def evaluate(self):
        self.training = False
