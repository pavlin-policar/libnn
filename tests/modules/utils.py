import numpy as np


def numeric_gradient(f, x, downstream_gradient=1, eps=1e-7):
    """Evaluate the numeric gradient for a given function.

    Parameters
    ----------
    f : Callable[[np.ndarray] -> np.ndarray]
        The funciton for which we evaluate the gradient.
    x : np.ndarray
        The point at which to evaluate the gradient at.
    downstream_gradient : Optional[np.ndarray]
        Observing the chain-rule dy/dx = dy/dz * dz/dx, the given function
        computes only dz/dx and dy/dz is passed downstream from
        backpropagation. If this is the last link in the chain, then dy/dy = 1,
        which is the default.
    eps : Optional[float]
        At what precision to evaluate the gradient at.

    Returns
    -------
    np.ndarray

    """
    x, grad = x.astype(np.float64), np.zeros_like(x, dtype=np.float64)

    for idx in np.ndindex(x.shape):
        initial_value = x[idx]

        x[idx] = initial_value + eps
        nudge_up = f(x)
        x[idx] = initial_value - eps
        nudge_down = f(x)
        # Restore x to its previous state
        x[idx] = initial_value

        gradient = ((nudge_up - nudge_down) * downstream_gradient) / (2 * eps)
        grad[idx] = np.sum(gradient)

    return grad
