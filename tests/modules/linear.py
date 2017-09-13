import unittest

import numpy as np

from libnn.modules.linear import Linear
from tests.modules.utils import numeric_gradient


class TestLinear(unittest.TestCase):
    def setUp(self):
        self.linear = Linear(4, 2)
        self.linear.W[:] = np.array([
            [1, 2],
            [2, 3],
            [3, 4],
            [4, 5],
        ], dtype=np.float64)

        self.linear.b[:] = np.ones(2, dtype=np.float64)

        self.matrix = np.array([
            [1, 1, 2, 2],
            [0, 0, 3, 3],
        ], dtype=np.float64)

        self.expected = np.array([
            [18, 24],
            [22, 28],
        ], dtype=np.float64)

        self.downstream_gradient = np.ones_like(self.expected, dtype=np.float64)

    def test_forward(self):
        np.testing.assert_equal(self.linear(self.matrix), self.expected)

    def test_backward_on_X(self):
        self.linear(self.matrix)
        d_X = self.linear.backward(self.downstream_gradient)

        np.testing.assert_almost_equal(
            numeric_gradient(self.linear.forward, self.matrix),
            d_X
        )

    def test_backward_on_W(self):
        self.linear(self.matrix)
        self.linear.backward(self.downstream_gradient)

        def forward_wrt_W(new_W):
            self.linear.W[:] = new_W
            return self.linear(self.matrix)

        np.testing.assert_almost_equal(
            numeric_gradient(forward_wrt_W, self.linear.W),
            self.linear.W.grad
        )

    def test_backward_on_b(self):
        self.linear(self.matrix)
        self.linear.backward(self.downstream_gradient)

        def forward_wrt_b(new_b):
            self.linear.b[:] = new_b
            return self.linear(self.matrix)

        np.testing.assert_almost_equal(
            numeric_gradient(forward_wrt_b, self.linear.b),
            self.linear.b.grad
        )
