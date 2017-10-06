import numpy as np

from libnn.modules import initializations
from libnn.modules.module import Module


class Linear(Module):
    def __init__(self, in_features, out_features, init='random_normal'):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        if callable(init):
            initialization = init
        elif hasattr(initializations, init):
            initialization = getattr(initializations, init)
        else:
            raise ValueError(
                '`%s` is not a recognized initialization scheme' % init
            )

        # For some initialization (e.g. xavier) to work properly, we must
        # initialize the weights at the same time. We add 1 to the
        # `in_features` to add to the bias term
        weights = initialization((in_features + 1, out_features))
        self.W = self.trainable(weights[:-1])
        self.b = self.trainable(weights[-1])

    def forward(self, X):
        self.save_for_backward(X)
        return X.dot(self.W) + self.b

    def backward(self, downstream_gradient):
        X, = self.saved_tensors
        self.W.grad = X.T.dot(downstream_gradient)
        self.b.grad = np.sum(downstream_gradient, axis=0)
        return downstream_gradient.dot(self.W.T)
