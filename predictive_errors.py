import numpy as np
from numpy import linalg as la
import LS_updates as LS

def pred_error(y, H, k, K, t, t0, var_y):

    Hk = H[:, :k+1]

    # START k x k
    Dk = la.inv(Hk[:t0, :k].T @ Hk[:t0, :k])
    theta_k = Dk @ Hk[:t0, :k].T @ y[:t0]

    Dtest = la.inv(Hk[:t0+1, :k].T @ Hk[:t0+1, :k])
    theta_test = Dtest @ Hk[:t0+1, :k].T @ y[:t0+1]

    Dkktest = la.inv(Hk[:t0+1, :k+1].T @ Hk[:t0+1, :k+1])
    thetakk_test = Dkktest @ Hk[:t0+1, :k+1].T @ y[:t0+1]

    # Update to k+1 x k+1
    theta_kk, Dkk, _ = LS.orls_ascend(y, H, k, K, 0, t0, Dk, theta_k)

    # Initialize
    G = 0
    THETA = theta_k

    # Time increments
    for i in range(t0, t):

        # Compute Ai
        A = Hk[i, :k] @ (Dkk[:k,k]/Dkk[k,k]) - Hk[i, k]
        #A = Hk[i, :k] @ Dk @ Hk[:i, :k].T @ Hk[:i, k] - Hk[i, k]

        # Compute Gi
        G = np.hstack((G, A*theta_kk[-1]))

        # Store THETAs using t0 elements
        THETA = np.hstack( [THETA, theta_k])

        # Update thetas
        theta_kk, Dkk = LS.trls_update(y[i], Hk[i, :k+1], theta_kk, Dkk, var_y)
        theta_k, Dk = LS.trls_update(y[i], Hk[i, :k], theta_k, Dk, var_y)

    # Residual error
    temp = Hk[t0:t, :k] * THETA[:,:][:,1:t-1].T
    E = y[t0:t] - np.sum(temp, axis=1).reshape(t-t0,1)

    return G[1:], E