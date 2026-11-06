import numpy as np

def l2_normalize(x: np.ndarray, axis: int = -1, eps: float = 1e-12) -> list:
    """
    L2-normalize x along the given axis.

    Args:
        x: input NumPy array
        axis: axis along which to normalize
        eps: small constant for numerical stability

    Returns:
        Normalized array as a nested Python list.
    """
    norms = np.maximum(np.linalg.norm(x, axis=axis, keepdims=True), eps) 
    xn = np.divide(x, norms) 
    return xn
    
