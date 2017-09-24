from functools import reduce

import numpy as np

from libnn.util import ForwardBackwardStorage


class TrainableParameter(np.ndarray):
    def __new__(cls, data, **kwargs):
        self = super().__new__(cls, data.shape, **kwargs)

        self[:] = data
        self.grad = None

        return self


class Module(ForwardBackwardStorage):
    def __init__(self):
        super().__init__()
        self._trainable_parameters = []
        self.training = True

    def trainable(self, *args, **kwargs):
        param = TrainableParameter(*args, **kwargs)
        self._trainable_parameters.append(param)
        return param

    def forward(self, X):
        raise NotImplementedError()

    def backward(self, downstream_gradient):
        raise NotImplementedError()

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def train(self):
        self.training = True

    def evaluate(self):
        self.training = False

    def trainable_parameters(self):
        yield from self._trainable_parameters


class Sequential(Module):
    def __init__(self, *modules):
        super().__init__()
        self._modules = modules

    def __len__(self):
        return len(self._modules)

    def forward(self, X):
        return reduce(lambda acc, layer: layer(acc), self._modules, X)

    def backward(self, downstream_gradient):
        return reduce(
            lambda acc, layer: layer.backward(acc),
            reversed(self._modules),
            downstream_gradient,
        )

    def trainable_parameters(self):
        for module in self._modules:
            yield from module.trainable_parameters()
