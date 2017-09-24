import numpy as np

from experiments import plotting
from libnn import datasets
from libnn.losses import BinaryCrossEntropy
from libnn.modules.activations import ReLU, Sigmoid
from libnn.modules.layers import Linear

X, y = datasets.xor_data(100)


linear1 = Linear(2, 50)
relu = ReLU()
linear2 = Linear(50, 1)
sigmoid = Sigmoid()
loss = BinaryCrossEntropy()


def model(X):
    z = linear1(X)
    z = relu(z)
    z = linear2(z)
    z = sigmoid(z)
    return z


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
    gradient = sigmoid.backward(gradient)
    gradient = linear2.backward(gradient)
    gradient = relu.backward(gradient)
    gradient = linear1.backward(gradient)

    # Update the weights
    linear1.W -= learning_rate * linear1.W.grad
    linear1.b -= learning_rate * linear1.b.grad

    linear2.W -= learning_rate * linear2.W.grad
    linear2.b -= learning_rate * linear2.b.grad

plotting.binary_decision_boundary(X, y, model)
