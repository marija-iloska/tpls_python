
# IMPORTS
import numpy as np
import matplotlib.pyplot as plt

''' Functions used often in the implementation

    get_min:  finds the index of the model with minimum predictive error
        and returns the corresponding quantities with that model
    
    get_features:  finds the location of the selected features 
        in the original feature matrix H
        
    initialize:  gives theta estimate with initial batch size
        and spares 1 data point to find first predictive error
        
    generate_data:  a function to generate synthetic data (output data without noise)
'''



# INDEX OF MODEL WITH MIN PREDICTIVE ERROR =======================================================
def get_min(theta: list, D: list, J: list, idx: list):

    # Find model with minimum predictive error
    minJ = J.index(min(J))

    # Return quantities associated with that model
    return theta[minJ], D[minJ], J[minJ], idx[minJ]




# FEATURE INDEX SORTING ===========================================================================
def get_features(H_true: np.ndarray, H_sorted: np.ndarray, K: int, k: int):
    
    ''' This functions finds the location of the selected features in the original
    feature matrix.

        K: Number of available features
        k: Number of features used
        H_true: Original feature matrix
        H_sorted: Feature matrix with reordered feature indices (from algorithm)
    '''

    # User input error
    if k > K:
        ValueError('Number of selected features must be less or equal to total available features.')
        
    # Create empty array to collect resorted features
    idx = np.array([], dtype=int)
    for i in range(K):
        if H_sorted[i] in H_true:
            idx = np.append(idx, np.where(H_true == H_sorted[i])[0])

    # Return selected or all resorted features
    return idx[:k]




# INITIAL ESTIMATE ================================================================================
def initialize(init_data: list):
    
    ''' This functions obtains theta estimate and predictive error
    with initial batch size of data. 

        init_data: A list that contains initial y, H, and feature indices
    '''

    # Data pair and initial features
    y0, H0, idx0 = init_data

    # Spare 1 data point for predictive error
    t0 = len(y0) - 1

    # CONDITIONS
    if t0 < len(idx0):
        ValueError('Number of data points must be greater than number of features used.')

    # Initialize parameter estimate
    D0 = np.linalg.inv(H0[:t0, idx0].T @ H0[:t0, idx0])
    theta0 = D0 @ H0[:t0, idx0].T @ y0[:t0]

    # Predictive error
    J0 = (y0[t0] - H0[t0, idx0] @ theta0) ** 2

    return theta0, D0, J0



# CREATE DATA FUNCTION =========================================================================
def generate_data(K: int, p: int, T: int, var_h: float, var_t: float):

    ''' This function creates output data y without noise,
        parameter theta, and feature matrix H.

        K: Number of available features
        p: Dimension of the true model (number of features used)
        T: Number of data points
        var_h: Noise varaince to create feature vectors
        var_t: Noise variance to create theta
    '''

    if p > K:
        ValueError('Total number of features must be greater than the number of features use.')

    # Create feature matrix
    H = np.random.normal(0, var_h, (T, K))

    # Create true parameter theta
    theta = np.random.normal(0, var_t, (K, 1))

    # Choose indices to set to 0
    all_idx = np.arange(K)
    idx = np.random.choice(all_idx, K-p, replace=False)
    theta[idx] = 0

    # Create data without noise
    y = H @ theta

    # Returns indices of nonzero values in theta

    return y, H, theta, np.setdiff1d(all_idx, idx)

    # FEATURES BAR PLOT =========================================================================
def bar_plot(correct: np.ndarray, incorrect: np.ndarray, t0: int, T: int, p: int, K:int):
    time_range = tuple(range(t0 + 1, T))

    # Number of features
    weight_counts = {
        "Correct": np.array(correct),
        "Incorrect": np.array(incorrect),
    }
    # Bar width
    width = 0.9

    fig, ax = plt.subplots()
    bottom = np.zeros(len(time_range))
    cols = ['mediumvioletred', 'lightgrey']
    i = 0
    for boolean, weight_count in weight_counts.items():
        pb = ax.bar(time_range, weight_count, width, label=boolean, bottom=bottom, color=cols[i])
        bottom += weight_count
        i += 1

    plt.axhline(y=p, color='black', linestyle='--', label='Horizontal Line', linewidth=3)
    ax.set_title("JPLS", fontsize=16)
    ax.set_xlabel('Time', fontsize=14)
    ax.set_ylabel('Number of Features', fontsize=14)
    ax.legend(loc="upper right")
    plt.ylim(0, K)
    plt.show()