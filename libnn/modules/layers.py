import numpy as np

from libnn.modules.module import Module


class Linear(Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        self.W = self.trainable(0.01 * np.random.randn(in_features, out_features))
        self.b = self.trainable(np.random.randn(out_features))

    def forward(self, X):
        self.save_for_backward(X)
        return X.dot(self.W) + self.b

    def backward(self, downstream_gradient):
        X, = self.saved_tensors
        self.W.grad = X.T.dot(downstream_gradient)
        self.b.grad = np.sum(downstream_gradient, axis=0)
        return downstream_gradient.dot(self.W.T)
