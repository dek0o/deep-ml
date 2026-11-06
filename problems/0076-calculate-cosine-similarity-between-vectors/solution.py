import numpy as np

def cosine_similarity(v1, v2) -> float:
	"""
	Calculate the cosine_similarity of two vectors.
	Args:
		vec1 (numpy.ndarray): 1D array representing the first vector.
		vec2 (numpy.ndarray): 1D array representing the second vector.
	Returns:
		The cosine_similarity of the two vectors.
	"""
	dot_product = 0
	sum_A_squares = 0 
	sum_B_squares = 0
	for i in range(len(v1)):
		dot_product += v1[i] * v2[i]
		sum_A_squares += v1[i]**2
		sum_B_squares += v2[i]**2
	
	return dot_product / (np.sqrt(sum_A_squares) * np.sqrt(sum_B_squares)) if sum_A_squares * sum_B_squares != 0 else 0