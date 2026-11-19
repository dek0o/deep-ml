import numpy as np

def to_categorical(x, n_col=None):
    # bring array 1D return array 2D
    n_col = n_col if n_col else max(x) + 1
    x_ohe = []
    for x_i in x:
        elem_ohe = [0] * n_col
        elem_ohe[x_i] = 1
        x_ohe.append(elem_ohe)
    return np.array(x_ohe)