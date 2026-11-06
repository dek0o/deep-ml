import numpy as np

def calculate_covariance_matrix(vectors: list[list[float]]) -> list[list[float]]:
	if len(vectors) == 1: return [[1.0]]
	means = [np.mean(feature) for feature in vectors]
	result_covs = []
	for i in range(len(vectors)):
		covs = []
		for j in range(len(vectors)):
			cov = sum([(x - means[i])*(y - means[j]) for x,y in zip(vectors[i], vectors[j])]) / (len(vectors[i]) - 1)
			covs.append(cov)
		result_covs.append(covs)
	return result_covs