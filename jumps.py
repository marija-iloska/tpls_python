import LS_updates as LS

# Jump down function
def down(y, H, var_y, theta, D, K, k, t, t0):

    # Initialize
    theta_store = D_store = idx_store = J_store = []

    # Loop through all models
    for m in range(k):

        # Update dimension down  k+1 ---> k
        theta, D, idx_H = LS.orls_descend(H, k, K, m, t, D, theta)

        # Compute PE  J(k+1,t) -- > J(k,t)
        J=2

        # Store
        theta_store.append(theta)
        D_store.append(D)
        idx_store.append(idx_H)
        J_store.append(J)

    # Find minimum predictive error
    minJ = J_store.index( min(J_store))

    # Update quantities
    theta = theta_store[minJ]
    idx_H = idx_store[minJ]
    J = J_store[minJ]
    D = D_store[minJ]
    k = k-1

    return theta, idx_H, J, D, k

# JUMP UP FUNCTION
def up(y, H, var_y, theta, D, K, k, t, t0):

    # Initialize
    theta_store = D_store = idx_store = J_store = []

    # Loop through all models
    for m in range(k):
        # Update dimension down  k+1 ---> k
        theta, D, idx_H = LS.orls_ascend(y, H, k, K, m, t, D, theta)

        # Compute PE  J(k+1,t) -- > J(k,t)
        J = 3

        # Store
        theta_store.append(theta)
        D_store.append(D)
        idx_store.append(idx_H)
        J_store.append(J)

    # Find minimum predictive error
    minJ = J_store.index(min(J_store))

    # Update quantities
    theta = theta_store[minJ]
    idx_H = idx_store[minJ]
    J = J_store[minJ]
    D = D_store[minJ]
    k = k + 1

    return theta, idx_H, J, D, k