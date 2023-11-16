from jpls import JPLS
from make_data import generate_data
import matplotlib as plt
import numpy as np

# DATA SETTINGS
T = 200  # Time series length
K = 7  # Total available features
p = 4  # True model order
var_y = 1  # Observation noise variance
var_h = 1  # Feature noise variance
var_t = 0.5  # Theta noise variance

# GENERATE DATA
y, H, theta, idx = generate_data(K, p, T, var_y, var_h, var_t)

# Initial batch size
t0 = K+1

# CALL JPLS
theta_k, idx_jpls, theta_store, J_pred = JPLS(y, H, t0, var_y)

idx = np.where(theta !=0)[0]
print(np.sort(idx_jpls[-1]))
print(list(np.sort(idx)))
