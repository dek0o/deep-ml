import numpy as np
def linear_regression_normal_equation(X: list[list[float]], y: list[float]) -> list[float]:
	np_X = np.array(X, dtype=float)
	np_y = np.array(y, dtype=float)
	# Your code here, make sure to round
	theta = np.linalg.inv(np_X.T @ np_X) @ np_X.T @ np_y
	return theta