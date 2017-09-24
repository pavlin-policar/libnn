import unittest
from functools import partial

import numpy as np

from libnn.losses import MeanSquaredError, BinaryCrossEntropy, \
    CategoricalCrossEntropy
from tests.modules.utils import numeric_gradient


class TestMeanSquaredError(unittest.TestCase):
    def test_cost(self):
        y_hat = np.array([4, 6, 5.5, 5])
        y = np.array([5, 5, 5, 5])

        mse = MeanSquaredError()
        self.assertEqual(mse(y_hat, y), 2.25 / 4)

    def test_gradient(self):
        y_hat = np.array([4, 6, 5.5, 5])
        y = np.array([5, 5, 5, 5])

        mse = MeanSquaredError()
        mse(y_hat, y)
        self.assertEqual(mse.gradient().shape, (4,))


class TestBinaryCrossEntropy(unittest.TestCase):
    def test_cost(self):
        y_hat = np.array([.8, .75, .25, .01, .99])
        y = np.array([1, 0, 0, 1, 1])

        bce = BinaryCrossEntropy()
        self.assertAlmostEqual(bce(y_hat, y), 1.3024, delta=1e-4)

    def test_gradient(self):
        y_hat = np.array([.8, .75, .25, .01, .99])
        y = np.array([1, 0, 0, 1, 1])

        bce = BinaryCrossEntropy()
        bce(y_hat, y)

        np.testing.assert_almost_equal(
            bce.gradient(),
            numeric_gradient(partial(bce, y=y), y_hat)
        )


class TestCategoricalCrossEntropy(unittest.TestCase):
    def test_cost(self):
        y_hat = np.array([
            [.2, .2, .2, .2, .2],
            [.2, .2, .2, .2, .2],
        ], dtype=np.float64)
        y = np.array([0, 3], dtype=np.float64)

        cce = CategoricalCrossEntropy()
        # Assuming all predictions are equally likely, then the loss should
        # equal to the log(n_classes)
        self.assertAlmostEqual(cce(y_hat, y), np.log(5), delta=1e-4)

    def test_gradient(self):
        y_hat = np.array([
            [.2, .2, .2, .2, .2],
            [.1, .1, .1, .6, .1],
            [.1, .1, .1, .1, .6],
        ], dtype=np.float64)
        y = np.array([0, 3, 4], dtype=np.float64)

        cce = CategoricalCrossEntropy()
        cce(y_hat, y)

        np.testing.assert_almost_equal(
            cce.gradient(),
            numeric_gradient(partial(cce, y=y), y_hat)
        )
