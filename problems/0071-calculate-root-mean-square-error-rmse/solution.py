import numpy as np

def rmse(y_true, y_pred):
	y_true = np.asarray(y_true)
	y_pred = np.asarray(y_pred)
	if y_true.shape != y_pred.shape: raise ValueError("")
	rmse_res = np.sqrt(np.mean((y_true - y_pred)**2))
	return round(rmse_res,3)
