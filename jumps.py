import LS_updates as LS
import predictive_errors as PE
import numpy as np

# JUMP UP FUNCTION
def up(y, H, theta_k, Dk, K, k, t, t0, J, var_y, e):

    # Initialize
    theta_store = []
    D_store = []
    idx_store = []
    J_store = []
    condition = []
    J_cond = []
    cond_log = []

    # Loop through all models
    for m in range(K-k):
        # Update dimension down  k+1 ---> k
        theta, D, idx_H, Hk = LS.orls_ascend(y, H, k, K, m, t, Dk, theta_k)

        # Compute PE  J(k+1,t) -- > J(k,t)
        G, E = PE.pred_error(y, Hk, k, K, t, t0, var_y)
        Jk = J + (G.T@G + 2*G.T@E)

        # expression on left
        cond = float( Hk[t-1,k]*(Hk[:t-1,k] @ y[:t-1])/(Hk[:t-1,k] @ Hk[:t-1,k]))

        # Store
        theta_store.append(theta)
        D_store.append(D)
        idx_store.append(idx_H)
        J_store.append(float(Jk[0]))
        condition.append( cond - 2*e < 0 )
        J_cond.append(Jk[0] < J)
        cond_log.append(cond - 2*e)


    # Find minimum predictive error
    minJ = J_store.index(min(J_store))

    # Update quantities
    theta = theta_store[minJ]
    idx_H = idx_store[minJ]
    J = J_store[minJ]
    D = D_store[minJ]
    k = k + 1

    return theta, idx_H, J, D, k


# JUMP DOWN FUNCTION
def down(y, H, theta_k, Dk, K, k, t, t0, J, var_y):

    # Initialize
    theta_store = []
    D_store = []
    idx_store = []
    J_store = []

    # Loop through all models
    for m in range(k):

        # Update dimension down  k+1 ---> k
        theta, D, idx_H, Hk = LS.orls_descend(H, k, K, m, t, Dk, theta_k)

        # Compute PE  J(k+1,t) -- > J(k,t)
        G,E = PE.pred_error(y, Hk, k-1, K, t, t0, var_y)
        Jk = J - (G.T@G + 2*G.T@E)

        # Store
        theta_store.append(theta)
        D_store.append(D)
        idx_store.append(idx_H)
        J_store.append(float(Jk[0]))

    # Find minimum predictive error
    minJ = J_store.index(min(J_store))

    # Update quantities
    theta = theta_store[minJ]
    idx_H = idx_store[minJ]
    J = J_store[minJ]
    D = D_store[minJ]
    k = k-1

    return theta, idx_H, J, D, k