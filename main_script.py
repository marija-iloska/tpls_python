import numpy as np
import LS_updates as LS
import jumps as JP
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


# CALL JPLS
def JPLS(y, H, var_y, t0):

    # Store true H
    H_true = H

    # Get dimensions
    T = len(H[:,0])
    K = len(H[0,:])

    # Initialize
    k = np.round(K/2)
    Hk = H[list(np.arange(init)), list(np.arange(k))]
    Dk = la.inv(Hk.T @ Hk)
    theta_k = Dk @ Hk.T @ y
    idx_H = list(range(k))

    # Initialize
    J = 0
    e = [0]
    J_pred = [0]
    J_jump = theta_jump = idx_jump = k_jump = Dk_jump = [0, 0, 0]
    theta_store = [theta_k]



    # Start time loop
    for t in range(t0+1,T):

        # Update J
        J_stay = J + np.power( (y[t-1] - Hk @ theta_k), 2)

        # Collect current state of estimate
        stay = [theta_k, list(range(k)), J, Dk, k]

        # MOVES =======================

        # STAY
        theta_jump[0], idx_jump[0], J_jump[0], Dk_jump[0], k_jump[0] = stay

        # JUMP UP
        if k < K:
            theta_jump[1], idx_jump[1], J_jump[1], Dk_jump[1], k_jump[1] = JP.up(y, H, var_y, theta_k, Dk, K, k, t, t0)
        else:
            J_jump[1] = float('inf')

        # JUMP DOWN
        if k > 1:
            theta_jump[2], idx_jump[2], J_jump[2], Dk_jump[2], k_jump[2] = JP.down(y, H, var_y, theta_k, Dk, K, k, t, t0)
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
        Hk = H[:t, list(range(k))]
        theta_store.append(theta_k)

        # Predictive error
        J_pred.append(J)
        e.append(y[t-1] - Hk @ theta_k)

        # FIND indices
        find_H = np.isin(H[0,:], H_true[0,:])
        idx_jpls = np.where(find_H == True)

        # TIME UPDATE
        theta_k, Dk = LS.trls_update(y, Hk[:,k-1], theta_k, Dk)

    return theta_store, J_pred

