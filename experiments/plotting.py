import matplotlib.pyplot as plt
import numpy as np


def scatter(X, y=None, indices=(0, 1), show=True):
    """Draw a scatter plot of the given dataset.

    Parameters
    ----------
    X : np.ndarray
    y : Optional[np.ndarray]
        The point labels, must be 1d and have same number of entries as X.
    indices : Optional[Tuple[int, int]]
        If `X` does not have 2 columns, then indices should be passed. These
        indices will be used as the x and y coordinate in the plot.
    show : Optional[bool]
        If the plot should be shown upon call.

    """
    assert X.ndim == 2, 'X must be 2 dimensional matrix'

    column_1, column_2 = indices
    assert column_1 < X.shape[1] and column_2 < X.shape[1], \
        'Index out of bounds for data with %d columns' % X.shape[1]

    plt.scatter(X[:, column_1], X[:, column_2], c=y, cmap=plt.cm.Paired,
                edgecolors='000', s=25)

    if show:
        plt.show()


def binary_decision_boundary(X, y, model, h=1e-2):
    """Draw a decision boundary for a binary classification model."""
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(
        np.arange(x_min, x_max, h),
        np.arange(y_min, y_max, h),
    )

    probs = model(np.c_[xx.ravel(), yy.ravel()])
    predictions = np.round(probs).astype(np.uint8)

    _decision_boundary(xx, yy, predictions)
    scatter(X, y, show=False)

    plt.show()


def decision_boundary(X, y, model, h=1e-2):
    """Draw a decision boundary for a multinomial classification model."""
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(
        np.arange(x_min, x_max, h),
        np.arange(y_min, y_max, h),
    )

    probs = model(np.c_[xx.ravel(), yy.ravel()])
    predictions = np.argmax(probs, axis=1)

    _decision_boundary(xx, yy, predictions)
    scatter(X, y, show=False)

    plt.show()


def _decision_boundary(xx, yy, predictions):
    """Draw the decision boundary colours on the global plot.

    Parameters
    ----------
    xx : np.ndarray
        X coordinates
    yy : np.ndarray
        y coordinates
    predictions : np.ndarray
        The predicted class label.
        
    """
    predictions = predictions.reshape(xx.shape)
    plt.contourf(xx, yy, predictions, cmap=plt.cm.Paired)
    plt.axis('off')
