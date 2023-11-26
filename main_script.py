from jpls import JPLS
from make_data import generate_data
import matplotlib as plt
import numpy as np
from Ep_test import compute_Ep
from random import sample

# DATA SETTINGS
T = 300  # Time series length
K = 7  # Total available features
p = 5  # True model order
var_y = 0.1  # Observation noise variance
var_h = 0.5  # Feature noise variance
var_t = 0.5  # Theta noise variance

# GENERATE DATA
y, H, theta, idx = generate_data(K, p, T, var_y, var_h, var_t)
idx = np.where(theta !=0)[0]
idx0 = idx[:p-1]
#idx0 = np.setdiff1d(idx, sample(list(idx), 1))
#idx1 = idx + sample(list(np.setdiff1d(np.arange(K), idx)), 1)
# Initial batch size
t0 = K+1

# CALL JPLS
theta_k, idx_jpls, theta_store, J_pred = JPLS(y, H, t0, var_y)


t = 250
sdiff, diff = compute_Ep(y, H, p, idx, t, var_y)

print(sdiff)
print(diff)

print(np.sort(idx_jpls[-1]))
print(list(np.sort(idx)))
