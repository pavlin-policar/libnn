import numpy as np

from libnn.losses import MeanSquaredError
from libnn.modules.activations import ReLU
from libnn.modules.layers import Linear
from libnn.modules.module import Sequential

np.random.seed(42)


x = np.array([
    [1, 2, 3, 4, 5],
    [2, 3, 4, 5, 6],
    [3, 4, 5, 6, 7],
])
y = np.array([1, 2, 3])

loss = MeanSquaredError()

model = Sequential(
    Linear(5, 30),
    ReLU(),
    Linear(30, 1),
)


learning_rate = 0.01
for epoch in range(500):
    # Perform the forward pass
    z = model(x)

    # Compute the error
    error = loss(z.flatten(), y)
    print('Epoch %2d: %.4f' % (epoch, error))

    # Backpropagate the gradient
    gradient = loss.gradient()[:, np.newaxis]
    gradient = model.backward(gradient)

    # Update the weights
    for parameter in model.trainable_parameters():
        parameter -= learning_rate * parameter.grad
