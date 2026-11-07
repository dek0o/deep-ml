import numpy as np

def numerical_gradient_check(f, x, analytical_grad, epsilon=1e-7):
    """
    Perform numerical gradient checking using centered finite differences.
    
    Args:
        f: A function that takes a numpy array and returns a scalar
        x: numpy array, the point at which to check gradient
        analytical_grad: numpy array, the analytically computed gradient
        epsilon: float, small value for finite difference approximation
    
    Returns:
        tuple: (numerical_grad, relative_error)
    """
    e = np.eye(len(x))
    numerical_grad = [(f(x + epsilon * e[i]) - f(x - epsilon * e[i])) / (2 * epsilon) for i in range(len(x))]

    rel_err = np.linalg.norm(numerical_grad - analytical_grad) / (np.linalg.norm(numerical_grad) + np.linalg.norm(analytical_grad))

    return tuple((numerical_grad, rel_err))