import numpy as np
from numpy import linalg as la
import LS_updates as LS
import jumps as JP

def JPLS(y, H, t0):

    # Store true H
    H_true = H

    # Get dimensions
    T = len(H[:,0])
    K = len(H[0,:])

    # Initialize
    k = round(K/2)
    Hk = H[list(range(t0)),:][:, list(range(k))]
    Dk = la.inv(Hk.T @ Hk)
    theta_k = Dk @ Hk.T @ y[:t0]
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
        theta_stay, idx_stay, J_stay, Dk_stay, k_stay = stay

        # JUMP UP
        if k < K:
            theta_up, idx_up, J_up, Dk_up, k_up = JP.up(y, H, theta_k, Dk, K, k, t, t0, J)

        else:
            J_up = float('inf')

        # JUMP DOWN
        if k > 1:
            theta_down, idx_down, J_down, Dk_down, k_down = JP.down(y, H, theta_k, Dk, K, k, t, t0, J)
        else:
            J_down = float('inf')

        print(t)
        # Store
        J_jump = [J_stay, J_up, J_down]
        k_jump = [k_stay, k_up, k_down]
        Dk_jump = [Dk_stay, Dk_up, Dk_down]
        idx_jump = [idx_stay, idx_up, idx_down]
        theta_jump = [theta_stay, theta_up, theta_down]


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
        theta_k, Dk = LS.trls_update(y[t-1], Hk[t-1,:k], theta_k, Dk)

    return theta_k, idx_jpls, theta_store, J_pred
