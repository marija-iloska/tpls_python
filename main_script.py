import numpy as np
from random import sample
from scipy import linalg


# FUNCTION THAT GENERATES DATA
def generate_data(K, p, T, var_y, var_h, var_t):
    H = np.random.normal(0, var_h, (T, K))
    theta = np.random.normal(0, var_t, (K, 1))
    idx = sample(list(range(1, K)), p)
    theta[idx] = 0
    y = np.dot(H, theta) + np.random.normal(0, var_y, (T, 1))
    return y, H, theta, idx


# DATA SETTINGS
T = 50  # Time series length
K = 5  # Total available features
p = 2  # True model order
var_y = 1  # Observation noise variance
var_h = 1  # Feature noise variance
var_t = 0.5  # Theta noise variance

# GENERATE DATA
y, H, theta, idx = generate_data(K, p, T, var_y, var_h, var_t)

D = linalg.inv(np.dot(H.T, H))


# CALL JPLS
def JPLS(y, H, idx, var_y, init):

    # Store true H
    H_true = H

    # Get dimensions
    T = len(H[:,1])
    K = len(H[1,:])

    # Initialize
    k = np.round(K/2)
    Hk = H[1:init, 1:k]
    Dk = linalg.inv(np.dot(Hk.T, Hk))
    theta_k = np.dot( np.dot(Dk, Hk.T), y)
    idx_H = range(k)

    # Initialize
    J = 0
    e = [0]
    J_pred = [0]
    J_jump = theta_jump = idx_jump = k_jump = Dk_jump = [0, 0, 0]
    theta_store = [theta_k]

    # Start time loop
    for t in range(T):

        # Update J
        J_stay = J + np.power( (y[t] - np.dot(H[t, 1:k], theta_k)), 2)

        # Collect current state of estimate
        stay = [theta_k, idx_H, J, Dk, k]

        # MOVES =======================

        # STAY
        theta_jump[0], idx_jump[0], J_jump[0], Dk_jump[0], k_jump[0] = stay

        # JUMP UP
        if k < K:
            theta_jump[1], idx_jump[1], J_jump[1], Dk_jump[1], k_jump[1] = stay
        else:
            J_jump[1] = float('inf')

        # JUMP DOWN
        if k > 1:
            theta_jump[2], idx_jump[2], J_jump[2], Dk_jump[2], k_jump[2] = stay
        else:
            J_jump[2] = float('inf')

        # CRITERION to jump
        minJ = J_jump.index( min(J_jump) )

        # Move to chosen model
        H = H[:, idx_jump[minJ]]
        J = J_jump[minJ]
        theta_k = theta_jump[minJ]
        Dk = Dk_jump[minJ]
        k = k_jump[minJ]

        # Some Quantity updates
        Hk = H[1:t, 1:k]
        theta_store.append(theta_k)

        # Predictive error
        J_pred.append(J)
        e.append(y[t] - np.dot(Hk, theta_k))

        # FIND indices
        find_H = np.isin(H[1,:], H_true[1,:])
        idx_H = np.where(find_H == True)

        # TIME UPDATE

    return

