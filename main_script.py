import util
import LS_updates
import algorithms
import numpy as np
import matplotlib as plt

# DATA SETTINGS
T = 300  # Time series length
K = 7  # Total available features
p = 5  # True model order
var_y = 0.1  # Observation noise variance
var_h = 0.5  # Feature noise variance
var_t = 0.5  # Theta noise variance

# SYNTHETIC DATA
# generate data and add gaussian noise
y, H, theta, idx = util.generate_data(K, p, T, var_h, var_t)
y = y + np.random.normal(0, var_y, (T, 1))



