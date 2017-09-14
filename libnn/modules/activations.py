import numpy as np

from libnn.modules.module import Module


class Activation:
    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def forward(self, X):
        ...

    def backward(self, x):
        ...


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

