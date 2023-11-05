import numpy as np
import scipy as sp
from random import sample

# DATA SETTINGS
T = 50       # Time series length
K = 5        # Total available features
p = 2        # True model order
var_y = 1    # Observation noise variance
var_h = 1    # Feature noise variance
var_t = 0.5  # Theta noise variance


# GENERATE DATA
#[y, H, theta] = generate_data(dy, p, T, var_y)
H = np.random.normal(0, var_h, (T, dy))
theta = np.random.normal(0, var_t, (dy, 1))
idx = sample(list(range(1,dy)), p)
theta[idx] = 0


# CALL JPLS


