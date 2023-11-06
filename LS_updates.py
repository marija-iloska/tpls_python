# ORLS ascending step


# ORLS descending step


# TRLS update
def time_update(y, hk, t, theta, var_y, D):
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
