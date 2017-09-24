import numpy as np

from experiments import plotting
from libnn import datasets
from libnn.losses import CategoricalCrossEntropy
from libnn.modules.activations import ReLU, Softmax
from libnn.modules.layers import Linear

X, y = datasets.spirals(100)


linear1 = Linear(2, 200)
relu = ReLU()
linear2 = Linear(200, 3)
softmax = Softmax()
loss = CategoricalCrossEntropy()


def model(X):
    z = linear1(X)
    z = relu(z)
    z = linear2(z)
    z = softmax(z)
    return z


learning_rate = 1
for epoch in range(100):
    # Perform the forward pass
    z = model(X)

    # Compute the error
    error = loss(z, y)
    print('Epoch %2d: %.4f' % (epoch, error))
    if np.isnan(error):
        break

    # Backpropagate the gradient
    gradient = loss.gradient()
    gradient = softmax.backward(gradient)
    gradient = linear2.backward(gradient)
    gradient = relu.backward(gradient)
    gradient = linear1.backward(gradient)

    # Update the weights
    linear1.W -= learning_rate * linear1.W.grad
    linear1.b -= learning_rate * linear1.b.grad

    linear2.W -= learning_rate * linear2.W.grad
    linear2.b -= learning_rate * linear2.b.grad

plotting.decision_boundary(X, y, model)
