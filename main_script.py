import util
import algorithms as alg
import numpy as np
import matplotlib.pyplot as plt

# DATA SETTINGS =======================================================
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


# JPLS ==================================================================
# INITIALIZE for JPLS
# Initial data size
t0 = 6

# Initial number of features to use
k = 3

# Indices of initial features used
init_idx = np.arange(k)
init_data = [y[:t0], H[:t0, :], init_idx]
theta_init, D_init, J_init = util.initialize(init_data)

# Pack variables
init_params = [theta_init, D_init, init_idx]

# Initialize instance of JPLS and update in time once
jfit = alg.jumpPLS(init_data, init_params, J_init, K, var_y)
jfit.time_update(y[t0-1], H[t0-1, init_idx])

# To store predictive error
J_pred = J_init

# Update with every new data point
for t in range(t0+1, T):

    # Receive new data point
    data_t = y[t]
    features_t = H[t, :]

    # JPLS model update step
    jfit.model_update(data_t, features_t, t)

    # JPLS time update step
    jfit.time_update(data_t, features_t[jfit.selected_features_idx])

    # Predictive Error store
    J_pred = np.append(J_pred, jfit.PredError)
    
    
    
plt.plot(J_pred)
plt.show()



