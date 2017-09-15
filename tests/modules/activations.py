import unittest

import numpy as np

from libnn.modules.activations import Softmax
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
        downstream_gradient = np.array([
            [1, 0, 0, 0, 0],
            [.2, .3, .3, .1, .1],
        ], dtype=np.float64)

        d_X = self.softmax.backward(downstream_gradient)
        print(d_X)

        np.testing.assert_almost_equal(
            numeric_gradient(self.softmax, downstream_gradient),
            d_X
        )
