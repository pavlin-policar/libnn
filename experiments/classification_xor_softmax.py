import numpy as np

from experiments import plotting
from libnn import datasets
from libnn.losses import CategoricalCrossEntropy
from libnn.modules.activations import ReLU, Softmax
from libnn.modules.layers import Linear

X, y = datasets.spirals(100)


loss = CategoricalCrossEntropy()

linear1 = Linear(2, 10)
relu1 = ReLU()
linear2 = Linear(10, 10)
relu2 = ReLU()
linear3 = Linear(10, 10)
relu3 = ReLU()
linear4 = Linear(10, 10)
relu4 = ReLU()
linear5 = Linear(10, 3)
softmax = Softmax()


def model(X):
    z = relu1(linear1(X))
    z = relu2(linear2(z))
    z = relu3(linear3(z))
    z = relu4(linear4(z))
    z = linear5(z)
    return softmax(z)


learning_rate = 0.1
for epoch in range(20000):
    # Perform the forward pass
    z = model(X)

    # Compute the error
    error = loss(z, y)

    # Backpropagate the gradient
    gradient = loss.gradient()
    gradient = softmax.backward(gradient)
    gradient = linear5.backward(gradient)
    gradient = relu4.backward(gradient)
    gradient = linear4.backward(gradient)
    gradient = relu3.backward(gradient)
    gradient = linear3.backward(gradient)
    gradient = relu2.backward(gradient)
    gradient = linear2.backward(gradient)
    gradient = relu1.backward(gradient)
    gradient = linear1.backward(gradient)
    print('Epoch %2d: %.4f [%.12f]' % (epoch, error, np.linalg.norm(gradient)))

    # Update the weights
    trainable_parameters = [
        parameter for layer in [linear1, linear2, linear3, linear4, linear5]
        for parameter in layer.trainable_parameters()
    ]
    for parameter in trainable_parameters:
        parameter -= learning_rate * parameter.grad

plotting.decision_boundary(X, y, model)
