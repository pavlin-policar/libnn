import matplotlib.pyplot as plt
import numpy as np

from libnn import datasets
from libnn.modules.activations import Softmax
from libnn.modules.linear import Linear


def scatter(X, y=None, indices=(0, 1), show=True):
    assert X.ndim == 2, 'X must be 2 dimensional matrix'

    column_1, column_2 = indices
    assert column_1 < X.shape[1] and column_2 < X.shape[1], \
        'Index out of bounds for data with %d columns' % X.shape[1]

    plt.scatter(X[:, column_1], X[:, column_2], c=y, cmap=plt.cm.Paired,
                edgecolors='000', s=25)

    if show:
        plt.show()


X, y = datasets.xor_data(100)
# scatter(X, y)


def softmax(x, axis=1):
    assert axis in (0, 1), 'Softmax only supports axis 0 and 1'

    z = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return z / np.sum(z, axis=axis, keepdims=True)


n_samples, n_attributes, n_classes = *X.shape, len(np.unique(y))
# Initialize theta to have `n_classes` columns and `n_attributes` rows
weights = np.zeros((n_attributes, n_classes))

l1 = Linear(n_attributes, n_classes)
l2 = Softmax()

# print(l2.forward(l1.forward(X)))


# weight_decay = 1e-7
# learning_rate = 1e-2
#
# for epoch in range(50):
#     probabilities = softmax(X.dot(weights))
#     # Compute the loss
#     data_loss = -np.sum(np.log(probabilities[np.arange(n_samples), y])) / n_samples
#     regularization_loss = 0.5 * weight_decay * np.sum(weights ** 2)
#     loss = data_loss + regularization_loss
#     # Compute the gradient
#     scores = probabilities
#     scores[np.arange(n_samples), y] -= 1
#     scores /= n_samples

#     data_grad = X.T.dot(scores)
#     regularization_grad = weight_decay * weights
#     grad = data_grad + regularization_grad

#     # Perform parameter update
#     weights -= learning_rate * grad
#     weights[0, :] = 0
#
#     print(loss)
