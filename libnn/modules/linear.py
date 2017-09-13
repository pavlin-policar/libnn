import numpy as np

from libnn.modules.module import Module


class Linear(Module):
    def __init__(self, in_features, out_features, add_bias=True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.add_bias = add_bias

        self.W = self.trainable(0.01 * np.random.randn(in_features, out_features))
        self.b = self.trainable(np.zeros((1, out_features)))

        self.__X_cache = None

    def forward(self, X):
        if self.training:
            self.__X_cache = X

        return X.dot(self.W) + self.b

    def backward(self, d_y):
        self.W.grad = self.__X_cache.T.dot(d_y)
        self.b.grad = np.sum(d_y, axis=0)
        return self.W.dot(d_y)
