import numpy as np


def numeric_gradient(f, x, eps=1e-6):
    x, grad = x.astype(np.float64), np.zeros_like(x, dtype=np.float64)

    for idx in np.ndindex(x.shape):
        initial_x = x[idx]

        x[idx] = initial_x + eps
        nudge_up = f(x)
        x[idx] = initial_x - eps
        nudge_down = f(x)
        # Restore x to its previous state
        x[idx] = initial_x

        centered_diff = (nudge_up - nudge_down) / (2 * eps)
        grad[idx] = np.sum(centered_diff)

    return grad
