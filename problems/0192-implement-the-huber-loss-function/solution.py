import numpy as np

def huber_loss(y_true, y_pred, delta=1.0):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    if y_true.shape != y_pred.shape:
        raise ValueError("y_true и y_pred должны быть одной формы")

    error = np.abs(y_true - y_pred)
    quadratic = 0.5 * error ** 2
    linear = delta * (error - 0.5 * delta)
    return float(np.mean(np.where(error <= delta, quadratic, linear)))