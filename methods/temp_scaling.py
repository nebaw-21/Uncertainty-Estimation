# Function for temperature scaling: Optimize scalar T on validation set to calibrate variances
# For regression, assume predictions are Gaussian (mean, var). Scale var by T to minimize NLL.
from scipy.optimize import minimize
import numpy as np

def temperature_scale(means_val, vars_val, y_val):
    # Negative log likelihood loss for Gaussian
    def nll_loss(T):
        T = np.abs(T[0])  # Ensure T is positive
        scaled_vars = vars_val / T
        nll = 0.5 * np.log(2 * np.pi * scaled_vars) + (y_val - means_val) ** 2 / (2 * scaled_vars)
        return np.mean(nll)
    res = minimize(nll_loss, x0=[1.0], bounds=[(1e-6, None)])
    return res.x[0]
