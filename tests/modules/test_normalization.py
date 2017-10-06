import unittest

import numpy as np

from libnn.modules.normalization import BatchNormalization
from tests.modules.utils import numeric_gradient


def _twos(*args):
    return 2. * np.ones(args)


class TestBatchNormalization(unittest.TestCase):
    def setUp(self):
        self.batch_norm = BatchNormalization(4, gamma_init='ones', beta_init='zeros')
        self.x = np.array([
            [2, 5, -2, 3],
            [4, -2, 1, 6],
            [-5, -2, 0, -2],
        ], dtype=np.float64)
        self.shape = self.x.shape[1]

    def test_forward_gamma(self):
        """Gamma parameter should scale std to value of gamma."""
        batch_norm = BatchNormalization(self.shape, gamma_init=_twos, beta_init='zeros')
        normalized = batch_norm(self.x)
        np.testing.assert_almost_equal(np.std(normalized, axis=0), 2.0, decimal=5)
        np.testing.assert_almost_equal(np.mean(normalized, axis=0), 0.0)

    def test_forward_with_beta(self):
        """Beta parameter should shift mean to value of beta."""
        batch_norm = BatchNormalization(self.shape, gamma_init='ones', beta_init=_twos)
        normalized = batch_norm(self.x)
        np.testing.assert_almost_equal(np.std(normalized, axis=0), 1.0, decimal=5)
        np.testing.assert_almost_equal(np.mean(normalized, axis=0), 2.0)

    def test_backward_on_X(self):
        downstream_gradient = np.ones_like(self.x)
        self.batch_norm(self.x)
        d_X = self.batch_norm.backward(downstream_gradient)

        np.testing.assert_almost_equal(
            d_X,
            numeric_gradient(self.batch_norm, self.x)
        )

    def test_backward_on_gamma(self):
        downstream_gradient = np.ones_like(self.x)
        self.batch_norm(self.x)
        self.batch_norm.backward(downstream_gradient)

        def _forward_wrt_gamma(new_gamma):
            self.batch_norm.gamma[:] = new_gamma
            return self.batch_norm(self.x)

        np.testing.assert_almost_equal(
            self.batch_norm.gamma.grad,
            numeric_gradient(_forward_wrt_gamma, self.batch_norm.gamma).ravel()
        )

    def test_backward_on_beta(self):
        downstream_gradient = np.ones_like(self.x)
        self.batch_norm(self.x)
        self.batch_norm.backward(downstream_gradient)

        def _forward_wrt_beta(new_beta):
            self.batch_norm.beta[:] = new_beta
            return self.batch_norm(self.x)

        np.testing.assert_almost_equal(
            self.batch_norm.beta.grad,
            numeric_gradient(_forward_wrt_beta, self.batch_norm.beta).ravel()
        )

