import unittest

import numpy as np

from libnn.losses import CategoricalCrossEntropy
from libnn.modules.activations import Softmax, ReLU, Sigmoid, Tanh, LeakyReLU, \
    PReLU, ELU
from tests.modules.utils import numeric_gradient


class TestSoftmax(unittest.TestCase):
    def setUp(self):
        self.softmax = Softmax()

    def test_forward(self):
        x = np.array([
            [2, 5, 6, 4, 3],
            [3, 4, 2, 2, 3]
        ], dtype=np.float64)
        z = self.softmax(x)

        # Softmax is monotone, so the arguments should not change order
        np.testing.assert_equal(np.argsort(x), np.argsort(z))
        # Make sure that row probabilities add up to 1
        np.testing.assert_almost_equal(np.sum(z, axis=1), np.ones(x.shape[0]))

    def test_backward(self):
        x = np.array([
            [2, 5, 6, 4, 3],
            [3, 2, 3, 1, 1],
        ], dtype=np.float64)
        downstream_gradient = np.ones_like(x)

        self.softmax(x)
        d_X = self.softmax.backward(downstream_gradient)

        np.testing.assert_almost_equal(
            numeric_gradient(self.softmax, x, downstream_gradient),
            d_X
        )

    def test_backward_with_simplified_cross_entropy_loss_gradient(self):
        """Compare the chained gradient with the known simplified gradient
        using cross-entropy loss."""
        x = np.array([
            [2, 5, 6, 4, 3],
            [3, 2, 3, 1, 1],
        ], dtype=np.float64)
        y = np.array([2, 4])

        y_hat = self.softmax(x)
        cce = CategoricalCrossEntropy()
        cce(y_hat, y)

        # Compute the softmax gradient wrt to loss using chained layers
        chained_gradient = self.softmax.backward(cce.gradient())

        # The gradient of the softmax simplifies greatly when using the
        # categorical cross entropy loss
        combined_gradient = y_hat
        combined_gradient[range(y.shape[0]), y] -= 1
        combined_gradient /= x.shape[0]

        np.testing.assert_almost_equal(chained_gradient, combined_gradient)


class TestReLU(unittest.TestCase):
    def setUp(self):
        self.relu = ReLU()

    def test_forward(self):
        x = np.array([
            [-2, 4, 1, 3, -1],
            [3, -2, 2, -3, 1],
        ], dtype=np.float64)
        expected = np.array([
            [0, 4, 1, 3, 0],
            [3, 0, 2, 0, 1],
        ], dtype=np.float64)

        np.testing.assert_equal(self.relu(x), expected)

    def test_backward(self):
        x = np.array([
            [-2, 4, 1, 3, -1],
            [3, -2, 2, -3, 1],
        ], dtype=np.float64)
        downstream_gradient = np.array([
            [5, -3, 1, 1, -3],
            [1, 2, -2, -2, 1],
        ], dtype=np.float64)

        self.relu(x)
        d_X = self.relu.backward(downstream_gradient)

        np.testing.assert_almost_equal(
            numeric_gradient(self.relu, x, downstream_gradient),
            d_X
        )


class TestSigmoid(unittest.TestCase):
    def setUp(self):
        self.sigmoid = Sigmoid()

    def test_forward(self):
        x = np.array([0, 1, -1])
        expected = np.array([.5, .731, .269])

        np.testing.assert_almost_equal(self.sigmoid(x), expected, decimal=3)

    def test_backward(self):
        x = np.array([0, 1, -1])
        downstream_gradient = np.ones_like(x) * 2

        self.sigmoid(x)
        d_X = self.sigmoid.backward(downstream_gradient)

        np.testing.assert_almost_equal(
            numeric_gradient(self.sigmoid, x, downstream_gradient),
            d_X
        )


class TestTanh(unittest.TestCase):
    def setUp(self):
        self.tanh = Tanh()

    def test_forward(self):
        x = np.array([0, 1, -1])
        expected = np.array([0., .762, -.762])

        np.testing.assert_almost_equal(self.tanh(x), expected, decimal=3)

    def test_backward(self):
        x = np.array([0, 1, -1])
        downstream_gradient = np.ones_like(x) * 2

        self.tanh(x)
        d_X = self.tanh.backward(downstream_gradient)

        np.testing.assert_almost_equal(
            numeric_gradient(self.tanh, x, downstream_gradient),
            d_X
        )


class TestLeakyReLU(unittest.TestCase):
    def setUp(self):
        self.leaky_relu = LeakyReLU(alpha=0.1)
        self.x = np.array([
            [-2, 4, 1, 3, -1],
            [3, -2, 2, -3, 1],
        ], dtype=np.float64)

    def test_forward(self):
        expected = np.array([
            [-0.2, 4, 1, 3, -0.1],
            [3, -0.2, 2, -0.3, 1],
        ], dtype=np.float64)

        np.testing.assert_almost_equal(self.leaky_relu(self.x), expected)

    def test_backward(self):
        downstream_gradient = np.array([
            [5, -3, 1, 1, -3],
            [1, 2, -2, -2, 1],
        ], dtype=np.float64)

        self.leaky_relu(self.x)
        d_X = self.leaky_relu.backward(downstream_gradient)

        np.testing.assert_almost_equal(
            d_X,
            numeric_gradient(self.leaky_relu, self.x, downstream_gradient)
        )


class TestPReLU(unittest.TestCase):
    def setUp(self):
        self.prelu = PReLU(alpha_init=lambda shape: 2 * np.ones(shape))
        self.x = np.array([
            [-2, 4, 1, 3, -1],
            [3, -2, 2, -3, 1],
        ], dtype=np.float64)

    def test_forward(self):
        expected = np.array([
            [-4, 4, 1, 3, -2],
            [3, -4, 2, -6, 1],
        ], dtype=np.float64)

        np.testing.assert_almost_equal(self.prelu(self.x), expected)

    def test_backward(self):
        downstream_gradient = np.array([
            [5, -3, 1, 1, -3],
            [1, 2, -2, -2, 1],
        ], dtype=np.float64)

        self.prelu(self.x)
        d_X = self.prelu.backward(downstream_gradient)

        np.testing.assert_almost_equal(
            d_X,
            numeric_gradient(self.prelu, self.x, downstream_gradient),
            decimal=5
        )

    def test_backward_wrt_alpha(self):
        downstream_gradient = np.array([
            [5, -3, 1, 1, -3],
            [1, 2, -2, -2, 1],
        ], dtype=np.float64)
        self.prelu(self.x)
        self.prelu.backward(downstream_gradient)

        def _forward_wrt_alpha(new_alpha):
            self.prelu.alpha[:] = new_alpha
            return self.prelu(self.x)

        np.testing.assert_almost_equal(
            self.prelu.alpha.grad,
            numeric_gradient(_forward_wrt_alpha, self.prelu.alpha)
        )


class TestELU(unittest.TestCase):
    def setUp(self):
        self.elu = ELU()
        self.x = np.array([
            [-2, 4, 1, 3, -1],
            [3, -2, 2, -3, 1],
        ], dtype=np.float64)

    def test_forward(self):
        x_mask = self.x > 0

        np.testing.assert_almost_equal(
            self.elu(self.x)[x_mask],
            self.x[x_mask]
        )
        self.assertFalse(np.any(np.equal(
            self.elu(self.x)[~x_mask],
            self.x[~x_mask]
        )))

    def test_backward(self):
        downstream_gradient = np.array([
            [5, -3, 1, 1, -3],
            [1, 2, -2, -2, 1],
        ], dtype=np.float64)

        self.elu(self.x)
        d_X = self.elu.backward(downstream_gradient)

        np.testing.assert_almost_equal(
            d_X,
            numeric_gradient(self.elu, self.x, downstream_gradient),
            decimal=5
        )
