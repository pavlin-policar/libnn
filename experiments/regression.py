import numpy as np

from libnn.losses import MeanSquaredError
from libnn.modules.layers import Linear
from libnn.modules.activations import ReLU

np.random.seed(42)


x = np.array([
    [1, 2, 3, 4, 5],
    [2, 3, 4, 5, 6],
    [3, 4, 5, 6, 7],
])
y = np.array([1, 2, 3])

linear1 = Linear(5, 3)
relu = ReLU()
linear2 = Linear(3, 1)
loss = MeanSquaredError()

learning_rate = 0.01
for epoch in range(50):
    # Perform the forward pass
    z = linear1(x)
    z = relu(z)
    z = linear2(z)

    # Compute the error
    error = loss(z.flatten(), y)
    print('Epoch %2d: %.4f' % (epoch, error))

    # Backpropagate the gradient
    gradient = loss.gradient()[:, np.newaxis]
    gradient = linear2.backward(gradient)
    gradient = relu.backward(gradient)
    gradient = linear1.backward(gradient)

    # Update the weights
    linear1.W -= learning_rate * linear1.W.grad
    linear1.b -= learning_rate * linear1.b.grad

    linear2.W -= learning_rate * linear2.W.grad
    linear2.b -= learning_rate * linear2.b.grad
