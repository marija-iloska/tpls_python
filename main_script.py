import numpy as np
import LS_updates as LS
from scipy import linalg as la
from make_data import generate_data

# DATA SETTINGS
T = 50  # Time series length
K = 6  # Total available features
p = 2  # True model order
var_y = 1  # Observation noise variance
var_h = 1  # Feature noise variance
var_t = 0.5  # Theta noise variance

# GENERATE DATA
y, H, theta, idx = generate_data(K, p, T, var_y, var_h, var_t)

D4 = la.inv(np.dot(H[:,:4].T, H[:,:4]))
theta4 = np.linalg.multi_dot([D4, H[:,:4].T, y])

D5or = la.inv(np.dot(H[:,:5].T, H[:,:5]))
theta5or = np.linalg.multi_dot([D5or, H[:,:5].T, y])

D3or = la.inv(np.dot(H[:,:3].T, H[:,:3]))
theta3or = np.linalg.multi_dot([D3or, H[:,:3].T, y])

theta3, D3, H3, idx_H3 = LS.orls_descend(H, len(theta4), K, 3, T, D4, theta4)
theta5, D5, H5, idx_H5 = LS.orls_ascend(y, H, len(theta4), K, 0, T, D4, theta4)


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

    return theta_store, J_pred

