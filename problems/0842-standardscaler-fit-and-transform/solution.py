import numpy as np

def fit(X_train: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    rows, cols = X_train.shape
    mean = np.array([])
    sigma = np.array([])
    for col in range(cols):
        mean = np.append(mean, np.mean(X_train[:, col]))
        s = np.std(X_train[:, col])
        sigma = np.append(sigma, s if s > 0 else 1)
    return mean, sigma

def transform(X_test: np.ndarray, means: np.ndarray, sigmas: np.ndarray) -> np.ndarray:
    rows, cols = X_test.shape
    for col in range(cols):
        X_test[:, col] = (X_test[:, col] - means[col]) / sigmas[col]
    return X_test

def standard_scaler(X_train: np.ndarray, X_test: np.ndarray) -> np.ndarray:
    """
    Fit a standard scaler on X_train and transform X_test.
    Returns the standardized X_test as a numpy array.
    """
    means, sigmas = fit(X_train)
    return transform(X_test, means, sigmas)

