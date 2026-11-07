import numpy as np

def gradient_descent(X, y, weights, learning_rate, n_epochs, batch_size=1, method='batch'):
    """
    Perform gradient descent optimization.
    
    Args:
        X: Feature matrix of shape (m, n)
        y: Target values of shape (m,)
        weights: Initial weights of shape (n,)
        learning_rate: Step size for gradient descent
        n_epochs: Number of complete passes through the dataset
        batch_size: Size of batches for mini-batch gradient descent (default: 1)
        method: Type of gradient descent ('batch', 'stochastic', or 'mini_batch')
    
    Returns:
        Optimized weights
    """
    m, n = X.shape
    y = y.reshape(-1, 1)
    theta = np.zeros((n, 1))
    if method == 'batch':
        while n_epochs != 0:
            y_pred = X @ theta
            grad = 2 / m * X.T @ (y_pred - y)
            theta = theta - learning_rate * grad
            n_epochs -= 1
    elif method == 'stochastic':
        while n_epochs != 0:
            for index in range(m):
                y_pred = X[index:index+1] @ theta
                grad = 2 * X[index:index+1].T @ (y_pred - y[index])
                theta = theta - learning_rate * grad
            n_epochs -= 1
    elif method == 'mini_batch':
        while n_epochs != 0:
            begin = 0
            end = batch_size
            for index in range(int(m // batch_size)):
                y_pred = X[begin:end] @ theta
                grad = 2/batch_size * X[begin:end].T @ (y_pred - y[begin:end])
                theta = theta - learning_rate * grad
                begin += batch_size
                end += batch_size
            n_epochs -= 1
    return theta.flatten()
