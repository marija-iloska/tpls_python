import numpy as np
from numpy import linalg as la
import LS_updates as LS
import jumps as JP


def JPLS(y, H, t0, var_y):
    # Store true H
    H_true = H

    # Get dimensions
    T = len(H[:, 0])
    K = len(H[0, :])

    # Initialize
    k = round(K / 2)
    Hk = H[list(range(t0 + 1)), :][:, list(range(k))]
    Dk = la.inv(Hk[:t0 - 1, :].T @ Hk[:t0 - 1, :])
    theta_k = Dk @ Hk[:t0 - 1, :].T @ y[:t0 - 1]
    idx_H = list(range(k))

    # Initialize
    e = y[t0 - 1] - Hk[t0 - 1, :] @ theta_k
    J = e ** 2
    e_pred = [e]
    J_pred = [J]
    theta_store = []
    idx_store = []
    # J_jump = theta_jump = idx_jump = k_jump = Dk_jump = [0, 0, 0]

    # Start time loop
    for t in range(t0, T - 1):

        # Update J stay
        e = y[t] - Hk[t, :] @ theta_k
        J = float(J + e ** 2)

        # Collect current state of estimate
        stay = [theta_k, list(range(K)), J, Dk, k]

        # MOVES =======================

        # STAY
        theta_stay, idx_stay, J_stay, Dk_stay, k_stay = stay

        # JUMP UP
        if k < K:
            theta_up, idx_up, J_up, Dk_up, k_up = JP.up(y, H, theta_k, Dk, K, k, t, t0, J, var_y)
        else:
            J_up = float('inf')

        # JUMP DOWN
        if k > 1:
            theta_down, idx_down, J_down, Dk_down, k_down = JP.down(y, H, theta_k, Dk, K, k, t, t0, J, var_y)
        else:
            J_down = float('inf')

        # Store
        J_jump = [J_stay, J_up, J_down]
        k_jump = [k_stay, k_up, k_down]
        Dk_jump = [Dk_stay, Dk_up, Dk_down]
        idx_jump = [idx_stay, idx_up, idx_down]
        theta_jump = [theta_stay, theta_up, theta_down]

        # CRITERION to jump
        minJ = J_jump.index(min(J_jump))

        # Move to chosen model
        H = H[:, idx_jump[minJ]]
        J = J_jump[minJ]
        theta_k = theta_jump[minJ]
        Dk = Dk_jump[minJ]
        k = k_jump[minJ]

        # Some Quantity updates
        Hk = H[:t + 2, list(range(k))]
        theta_store.append(theta_k)

        # Predictive error
        J_pred.append(J)
        e_pred.append(y[t] - Hk[t, :] @ theta_k)

        # FIND which features were used
        idx_jpls = []
        for i in range(K):
            if H[0, i] in H_true[0, :]:
                idx_jpls.append(int(np.where(H_true[0, :] == H[0, i])[0]))
        idx_store.append(idx_jpls[:k])

        # TIME UPDATE
        theta_k, Dk = LS.trls_update(y[t], Hk[t, :k], theta_k, Dk, var_y)

    return theta_k, idx_store, theta_store, J_pred
