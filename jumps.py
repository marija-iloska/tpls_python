import LS_updates as LS
import predictive_errors as PE
import numpy as np

# Jump down function
def down(y, H, theta_k, Dk, K, k, t, t0, J):

    # Initialize
    theta_store = np.empty((k-1,1))
    D_store = np.empty((k-1,k-1))
    idx_store = np.arange(K)
    J_store = np.array([float('inf')])

    # Loop through all models
    for m in range(k):

        # Update dimension down  k+1 ---> k
        theta, D, idx_H, idx_k = LS.orls_descend(H, k, K, m, t, Dk, theta_k)

        # Compute PE  J(k+1,t) -- > J(k,t)
        G,E = PE.pred_error(y, H[:,idx_k], k, K, t, t0)
        Jk = J - (G @ G.T + 2 * G @ E)

        print(J_store)
        # Store
        theta_store = np.hstack((theta_store, theta))
        D_store = np.hstack((D_store, D))
        idx_store = np.vstack((idx_store, idx_H))
        J_store = np.vstack((J_store, Jk))

    # Find minimum predictive error
    minJ = np.where(J_store == min(J_store))
    minJ = int(minJ[0]) - 1

    # Update quantities
    theta = theta_store[:,minJ + 1].reshape(k-1,1)
    idx_H = idx_store[minJ + 1,:]
    J = J_store[minJ+1]
    D = D_store[:, k -1 + minJ*(k-1) : (minJ+1)*(k-1) +k -1]
    k = k-1

    return theta, idx_H, J, D, k

# JUMP UP FUNCTION
def up(y, H, theta_k, Dk, K, k, t, t0, J):

    # Initialize
    theta_store = np.empty((k+1,1))
    D_store = np.empty((k+1,k+1))
    idx_store = np.arange(K)
    J_store = np.array([float('inf')])

    # Loop through all models
    for m in range(k):
        # Update dimension down  k+1 ---> k
        theta, D, idx_H = LS.orls_ascend(y, H, k, K, m, t, Dk, theta_k)

        # Compute PE  J(k+1,t) -- > J(k,t)
        G, E = PE.pred_error(y, H[:,idx_H], k, K, t, t0)
        Jk = J + (G @ G.T + 2 * G @ E)

        print(J_store)

        # Store
        theta_store = np.hstack((theta_store, theta))
        D_store = np.hstack((D_store, D))
        idx_store = np.vstack((idx_store, idx_H))
        J_store = np.hstack((J_store, Jk))

    # Find minimum predictive error
    minJ = np.where(J_store == min(J_store))
    minJ = int(minJ[0]) -1

    # Update quantities
    theta = theta_store[:,minJ + 1].reshape(k+1,1)
    idx_H = idx_store[minJ + 1,:]
    J = J_store[minJ + 1]
    D = D_store[:,  k+1 + minJ*(k+1): (minJ+1)*(k+1) + k+1]
    k = k + 1

    return theta, idx_H, J, D, k