import numpy as np

from experiments import plotting
from libnn import datasets
from libnn.losses import BinaryCrossEntropy
from libnn.modules.activations import ReLU, Sigmoid
from libnn.modules.layers import Linear
from libnn.modules.module import Sequential

X, y = datasets.xor_data(100)


loss = BinaryCrossEntropy()

model = Sequential(
    Linear(2, 50),
    ReLU(),
    Linear(50, 1),
    Sigmoid(),
)


learning_rate = 1
for epoch in range(1000):
    # Perform the forward pass
    z = model(X)

    # Compute the error
    error = loss(z.flatten(), y)
    print('Epoch %2d: %.4f' % (epoch, error))
    if np.isnan(error):
        break

    # Backpropagate the gradient
    gradient = loss.gradient()[:, np.newaxis]
    gradient = model.backward(gradient)

    # Update the weights
    for parameter in model.trainable_parameters():
        parameter -= learning_rate * parameter.grad

plotting.binary_decision_boundary(X, y, model)
