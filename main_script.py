import util
import algorithms as alg
import numpy as np
import matplotlib.pyplot as plt
import LS_updates as ls

# DATA SETTINGS =======================================================
T = 700  # Time series length
K = 8  # Total available features
p = 4  # True model order
var_y = 2  # Observation noise variance
var_h = 1  # Feature noise variance
var_t = 0.5  # Theta noise variance

# SYNTHETIC DATA
# generate data and add gaussian noise
y, H, theta, idx = util.generate_data(K, p, T, var_h, var_t)
y = y + np.random.normal(0, var_y, (T, 1))


# JPLS ==================================================================
# INITIALIZE for JPLS
t0 = 15  # Initial data size
k = 3   # Initial number of features to use

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
missing = []
correct = []
incorrect = []

# Update with every new data point
for t in range(t0+1, T):

    # Receive new data point
    data_t = y[t]
    features_t = H[t, :]

    # JPLS model update step
    jfit.model_update(data_t, features_t, t)

    # JPLS time update step
    idx_jpls = jfit.selected_features_idx
    jfit.time_update(data_t, features_t[idx_jpls])

    # Store Results
    correct.append(np.sum(np.isin(idx_jpls, idx)))
    incorrect.append(jfit.k - correct[-1])
    missing.append(p - correct[-1])

    # Predictive Error store
    J_pred = np.append(J_pred, jfit.PredError)
    

mse = ls.Expectations(y, H, t0, T, K, idx, var_y)

Ed = mse.model_down()
Eu = mse.model_up()

Edt = mse.batch(Ed)
Eut = mse.batch(Eu)


for i in range(Eu.shape[0]):
    plt.plot( np.arange(t0+1,T), Eu[i, :])
plt.show()

for i in range(Ed.shape[0]):
    plt.plot( np.arange(t0+1,T), Ed[i, :])
plt.show()


plt.plot(J_pred, linewidth=3, color='#A52A2A', label='JPLS')
plt.xlabel('Time', fontsize=14)
plt.ylabel('Predictive Error', fontsize=14)
#plt.title('Results', fontsize=16)
plt.legend(fontsize=12)

plt.show()


jpls = np.array([correct, incorrect])


import matplotlib.pyplot as plt
import numpy as np


time_range = tuple(range(t0+1, T))

weight_counts = {
    "Correct": np.array(correct),
    "Incorrect": np.array(incorrect),
}
width = 0.9

fig, ax = plt.subplots()
bottom = np.zeros(len(time_range))
cols = ['mediumvioletred', 'lightgrey']
i = 0
for boolean, weight_count in weight_counts.items():
    pb = ax.bar(time_range, weight_count, width, label=boolean, bottom=bottom, color = cols[i])
    bottom += weight_count
    i+=1

plt.axhline(y=p, color='black', linestyle='--', label='Horizontal Line', linewidth = 3)
ax.set_title("JPLS", fontsize = 16)
ax.set_xlabel('Time', fontsize = 14)
ax.set_ylabel('Number of Features', fontsize = 14)
ax.legend(loc="upper right")
plt.ylim(0, K)

plt.show()


plt.plot(J_pred)
plt.show()