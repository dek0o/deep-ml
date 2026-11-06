import numpy as np

def matrixmul(a:list[list[int|float]],
              b:list[list[int|float]])-> list[list[int|float]]:
	return np.asarray(a) @ np.asarray(b) if len(a[0]) == len(b) else -1