import numpy as np

def sgd_update(X: np.ndarray, y: np.ndarray, weights: np.ndarray, learning_rate: float, n_iter: int) -> list:
    """
    Perform n_iter steps of stochastic gradient descent on a linear regression
    model with MSE loss, cycling through samples in order.

    Returns the final weight vector as a Python list.
    """
    m, n = np.shape(X)
    for i in range(n_iter):
        sample_idx = i % m
        y_pred = X[sample_idx] @ weights # (n, ) @ (n, ) = dot product
        grad = 2 * X[sample_idx] * (y_pred - y[sample_idx]) # (n, ) @ (1, 1) = (n, 1)
        weights = weights - learning_rate * grad
    return weights