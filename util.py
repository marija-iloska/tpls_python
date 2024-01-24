
''' Functions used often in the implementation

    get_min:  finds the index of the model with minimum predictive error
        and returns the corresponding quantities with that model
    
    get_features:  finds the location of the selected features 
        in the original feature matrix H
        
    initialize:  gives theta estimate with initial batch size
        and spares 1 data point to find first predictive error
        
    generate_data:  a function to generate synthetic data
'''



# INDEX OF MODEL WITH MIN PREDICTIVE ERROR
def get_min(theta, D, J, idx):
    # Find model with minimum predictive error
    minJ = J.index(min(J))

    # Return quantities associated with that model
    return theta[minJ], D[minJ], J[minJ], idx[minJ]


# FEATURE INDEX SORTING
def get_features(H_true, H_sorted, K, k):

    # Create empty array to collect resorted features
    idx = np.array([], dtype=int)
    for i in range(K):
        if H_sorted[i] in H_true:
            idx = np.append(idx, np.where(H_true == H_sorted[i])[0])
    # Return selected or all resorted features
    return idx[:k]


def initialize(init_data):

    # Data pair and initial features
    y0, H0, idx0 = init_data

    # Spare 1 data point for predictive error
    t0 = len(y0) - 1

    # Initialize parameter estimate
    D0 = la.inv(H0[:t0, idx0].T @ H0[:t0, idx0])
    theta0 = D0 @ H0[:t0, idx0].T @ y0[:t0]

    # Predictive error
    J0 = (y0[t0] - H0[t0, idx0] @ theta0) ** 2

    return theta0, D0, J0


# CREATE DATA FUNCTION
def generate_data(K, p, T, var_y, var_h, var_t):

    ''' This functions create output data y without noise,
        parameter theta, and feature matrix H. '''

    # Create feature matrix
    H = np.random.normal(0, var_h, (T, K))

    # Create true parameter theta
    theta = np.random.normal(0, var_t, (K, 1))

    # Choose indices to set to 0
    idx = sample(list(range(1, K)), K - p)
    theta[idx] = 0

    # Create data without noise
    y = np.dot(H, theta)

    return y, H, theta, idx

