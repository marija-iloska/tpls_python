

# CREATE DATA FUNCTION
def generate_data(K, p, T, var_y, var_h, var_t):
    H = np.random.normal(0, var_h, (T, K))
    theta = np.random.normal(0, var_t, (K, 1))
    idx = sample(list(range(1, K)), p)
    theta[idx] = 0
    y = np.dot(H, theta) + np.random.normal(0, var_y, (T, 1))
    return y, H, theta, idx
