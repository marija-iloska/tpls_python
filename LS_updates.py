import numpy as np

# ORLS ascending ====================================
def orls_ascend(y, H, k, K, m, t, D, theta):

    # Current input data (loops 0 to index minus 1)
    Hk = H[:t, :k]
    y_past = y[:t]

    # Define the new feature h_new(k+1)
    h_new = H[:t, k+m]

    # Projection matrix
    DHkT = D @ Hk.T
    P_norm = np.eye(t) - Hk @ DHkT


    # Some reusable terms
    hnP = h_new @ P_norm
    hnPy = hnP @ y_past
    d = DHkT @ h_new


    # Compute terms of D(k+1)
    D22 = 1 / (hnP @ h_new)
    D12 = - d*D22
    D21 = D12.T
    D11 = D + np.outer(d,d)*D22

    # Create D(k+1)
    top = np.vstack([D11, D12])
    D = np.hstack([top, np.append(D21, D22).reshape(k+1,1)])

    # Update theta(k) to theta(k+1)
    hnPyD = hnPy*D22  #reusable term
    theta = np.vstack( [theta - hnPyD*d.reshape(k,1), hnPyD])

    # Update feature indices
    idx_H = list(range(0,k)) + [k+m] + list( np.setdiff1d( list(range(k,K)), [k+m] ) )

    # Update Hk in time and order
    Hk = H[0:t+1,:][:, idx_H[:k+1]]


    return theta, D, idx_H, Hk


# ORLS descending step ====================================
def orls_descend(H, k, K, m, t, D, theta):

    # Range of indices to include
    idx = list(np.setdiff1d( list(range(0,k)),m ))

    # Update Hk
    Hk = H[:t+1,:][:, idx + [m]]

    # Update H
    idx_H = idx + list(range(k,K)) + [m]

    # Get D(k+1) bar
    Dswap = np.empty((k, k))
    Dswap[0:k-1,:][:,0:k-1] = D[idx, :][:, idx]
    Dswap[k-1,0:k-1] = D[m, idx]
    Dswap[:, k-1] = D[ idx+[m], m]

    # Get D(k+1) bar blocks
    D11 = Dswap[:k-1,:][:,:k-1]
    D12 = Dswap[:k-1, k-1:]
    D21 = D12.T
    D22 = Dswap[k-1,k-1]

    # Get D(k) final
    D = D11 - (D12 @ D21)/D22

    # Ratio
    ratio = D12/D22

    # Update rest of theta
    theta = theta[idx] - theta[m]*ratio.reshape(k-1,1)

    idx_k = idx + [m] + list(range(k,K))

    return theta, D,  idx_H, Hk



# TRLS update ====================================
def trls_update(y, hk, theta, D, var_y):
    # System dimension
    k = len(theta)

    # Current error
    e = y - hk @ theta

    # Find current sigma
    Sigma = var_y*D

    # Update gain
    temp = Sigma @ hk
    K = temp / (var_y + hk @ temp)

    # Update estimate
    theta = theta + e*K.reshape(k,1)

    # Update covariance
    temp = np.eye(k) - np.outer(K, hk)
    Sigma = np.matmul(temp, Sigma)

    # Update D
    D = Sigma/var_y

    return theta, D
