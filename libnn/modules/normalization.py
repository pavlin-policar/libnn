import numpy as np

from libnn.modules import initializations
from libnn.modules.module import Module


class BatchNormalization(Module):
    def __init__(
        self,
        in_shape,
        epsilon=1e-6,
        gamma_init='random_normal',
        beta_init='random_normal'
    ):
        super().__init__()

        # Gamma initialization scheme
        if callable(gamma_init):
            gamma_initialization = gamma_init
        elif hasattr(initializations, gamma_init):
            gamma_initialization = getattr(initializations, gamma_init)
        else:
            raise ValueError(
                '`%s` is not a recognized initialization scheme' % gamma_init
            )

        # Beta initialization scheme
        if callable(beta_init):
            beta_initialization = beta_init
        elif hasattr(initializations, beta_init):
            beta_initialization = getattr(initializations, beta_init)
        else:
            raise ValueError(
                '`%s` is not a recognized initialization scheme' % beta_init
            )

        self.gamma = self.trainable(gamma_initialization(1, in_shape))
        self.beta = self.trainable(beta_initialization(1, in_shape))
        self.epsilon = epsilon

    def forward(self, X):
        # TODO Normalize across feature map after convolutional layers
        # TODO Store std/mean as running averages during training to use at test time 1:04
        if self.training:
            mean, var = np.mean(X, axis=0), np.var(X, axis=0)
        elif self.running_mean is None or self.running_var is None:
            raise RuntimeError(
                'Running mean/var not defined, please train model first.'
            )
        else:
            mean, var = self.running_mean, self.running_var

        inv_std = 1. / np.sqrt(var + self.epsilon)
        x_hat = (X - mean) * inv_std
        self.save_for_backward(inv_std, x_hat)

        return self.gamma * x_hat + self.beta

    def backward(self, downstream_gradient):
        dx_hat = downstream_gradient * self.gamma
        N = downstream_gradient.shape[0]

        inv_std, x_hat = self.saved_tensors

        self.beta.grad = np.sum(downstream_gradient, axis=0)
        self.gamma.grad = np.sum(x_hat * downstream_gradient, axis=0)
        return (1 / N) * inv_std * (
            N * dx_hat - np.sum(dx_hat, axis=0) -
            x_hat * np.sum(dx_hat * x_hat, axis=0)
        )
