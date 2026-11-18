import numpy as np
from collections import Counter

def stratified_train_test_split(X, y, test_size, random_seed=None):
    """
    Split data into train and test sets while maintaining class proportions.
    
    Args:
        X: Feature matrix of shape (n_samples, n_features)
        y: Label vector of shape (n_samples,)
        test_size: Proportion of data for test set (0 < test_size < 1)
        random_seed: Random seed for reproducibility
    
    Returns:
        X_train, X_test, y_train, y_test
    """
    n,m = X.shape
    np.random.seed(random_seed)
    n_samples = int(n * test_size)

    indices = list(range(n))
    np.random.shuffle(indices)

    classes = Counter(y)
    classes_test_size = {y: int(count * test_size) for y, count in classes.items()}
    test_indices = []
    for i in indices:
        if classes_test_size[y[i]] > 0:
            test_indices.append(i)
            classes_test_size[y[i]] = classes_test_size.get(y[i], 0) - 1
    train_indices = []
    for i in indices:
        if i not in test_indices:
            train_indices.append(i)
    return X[train_indices], X[test_indices], y[train_indices], y[test_indices]