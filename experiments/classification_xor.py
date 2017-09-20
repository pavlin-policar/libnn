import matplotlib.pyplot as plt
import numpy as np

from libnn import datasets
from libnn.modules.activations import ReLU, Sigmoid
from libnn.modules.layers import Linear
from libnn.losses import BinaryCrossEntropy


def scatter(X, y=None, indices=(0, 1), show=True):
    assert X.ndim == 2, 'X must be 2 dimensional matrix'

    column_1, column_2 = indices
    assert column_1 < X.shape[1] and column_2 < X.shape[1], \
        'Index out of bounds for data with %d columns' % X.shape[1]

    plt.scatter(X[:, column_1], X[:, column_2], c=y, cmap=plt.cm.Paired,
                edgecolors='000', s=25)

    if show:
        plt.show()


def binary_decision_boundary(X, y, model, h=1e-2):
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(
        np.arange(x_min, x_max, h),
        np.arange(y_min, y_max, h),
    )

    probs = model(np.c_[xx.ravel(), yy.ravel()])
    predictions = np.round(probs).astype(np.uint8)

    predictions = predictions.reshape(xx.shape)
    plt.contourf(xx, yy, predictions, cmap=plt.cm.Pastel2)
    plt.axis('off')

    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.Paired)
    plt.show()


X, y = datasets.xor_data(100)
# scatter(X, y)


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

binary_decision_boundary(X, y, model)
