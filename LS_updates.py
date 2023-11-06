# ORLS ascending step
def orls_update(y, H, k, K, m, t, D, theta):

    # Current input data
    Hk = H[1:(t-1), 1:k]
    y_past = y[1:(t-1)]

    # Define the new feature h_new(k+1)
    h_new = H[1:(t-1), k + m]

    # Projection matrix
    DHkT = np.dot(D, Hk.T)
    P_norm = np.eye(t-1) - np.dot(Hk, DHkT)


    # Some reusable terms
    hnP = np.dot(h_new.T, P_norm)
    hnPy = np.dot(hnP, y_past)
    d = np.dot(DHkT, h_new)


    # Compute terms of D(k+1)
    D22 = 1 / np.dot(hnP, h_new)
    D12 = - d*D22
    D21 = D12.T
    D11 = D + D22*np.dot(d,d.T)

    # Create D(k+1)
    D = np.array([[D11, D12], [D21, D22]])

    # Update theta(k) to theta(k+1)
    hnPyD = hnPy*D22  #reusable term
    theta = [theta - np.dot(d, hnPyD), hnPyD]

    # Update Hk in time and order
    Hk = H[1:t, [1:k, k+m]]

    # Update feature indices
    idx_H = [1:k, k+m, np.setdiff1d( [(k+1):K],[k+m] ) ]

    return theta, D, Hk, idx_H




# ORLS descending step


# TRLS update
def trls_update(y, hk, t, theta, var_y, D):
    # System dimension
    k = len(theta)
    # Current error
    e = y - np.dot(theta, hk)

    # Update gain
    temp = np.dot(Sigma, hk)
    K = temp / (var_y + np.dot(hk.T, temp))

    # Update estimate
    theta = theta + np.dot(K, e)

    # Update covariance
    temp = (np.eye(k) - np.dot(hk.T, K))
    D = np.matmul(temp, D)

    return theta, D
