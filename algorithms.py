import numpy as np
import LS_updates as ls
import util

# SCAN OVER ALL FEATURES ================================================================================
class ModelJump:

    ''' Computes and stores quantities in the cases of:
      Jumping a model dimension up  k --> k + 1
      Jumping a model dimension down k --> k - 1
      Remaining in the same model k --> k '''

    def __init__(self, theta: np.ndarray, D: np.ndarray, y: np.ndarray, H: np.ndarray, K: int, var_y: float,
                 J: np.ndarray, times: tuple):

        self.J = J
        self.theta = theta
        self.D = D
        self.model = ls.ORLS(theta, D, y, H, K)
        self.PE = ls.PredictiveError(y, times[0], times[1], K, var_y)


    # UP -----------------------------------------------------------------------
    def up(self):

        ''' This function computes and stores the predictive error
        for adding any one feature from the unused available features.  '''

        model = self.model
        PE = self.PE

        # Initialize
        theta_store = []
        D_store = []
        idx_store = []
        J_store = []

        # Loop through all models
        for m in range(model.K - model.k):
            # Update dimension down  k+1 ---> k
            theta, D, idx_H, Hk = model.ascend(m)

            # Compute PE  J(k+1,t) -- > J(k,t)
            G, E = PE.compute(Hk, model.k)
            Jk = self.J + (G.T @ G + 2 * G.T @ E)

            # Store
            theta_store.append(theta)
            D_store.append(D)
            idx_store.append(idx_H)
            J_store.append(Jk)

        theta, D, J, idx_H = util.get_min(theta_store, D_store, J_store, idx_store)

        return theta, idx_H, J, D


    # DOWN -----------------------------------------------------------------------
    def down(self):

        ''' This function computes and stores the predictive error
        for removing any one feature from the current model.  '''

        model = self.model
        PE = self.PE

        # Initialize
        theta_store = []
        D_store = []
        idx_store = []
        J_store = []

        # Loop through all models
        for m in range(model.k):
            # Update dimension down  k+1 ---> k
            theta, D, idx_H, Hk = model.descend(m)

            # Compute PE  J(k+1,t) -- > J(k,t)
            G, E = PE.compute(Hk, model.k - 1)
            Jk = self.J - (G.T @ G + 2 * G.T @ E)

            # Store
            theta_store.append(theta)
            D_store.append(D)
            idx_store.append(idx_H)
            J_store.append(Jk)

        theta, D, J, idx_H = util.get_min(theta_store, D_store, J_store, idx_store)

        return theta, idx_H, J, D

    # STAY -----------------------------------------------------------------------
    def stay(self):
        ''' This function stores all the quantities
        for remaining in the present model. '''

        idx_H = np.arange(self.model.K)
        return self.theta, idx_H, self.J, self.D


# JUMP PREDICTIVE LEAST SQUARES - One time step in ALGORITHM 4 ======================================================
class JPLS:
    
    ''' Algorithm step that finds best model as we collect data point.
    Uses ModelJump class to scan over features, get predictive error and decide model

    model_update: uses ORLS to add/remove a feature or to stay
    time_update: uses RLS to update with new data point
    '''

    def __init__(self, initials: list, params: list, predictive_error: float, num_available_features: int,
                 noise_variance: float):

        '''
         initials: initial data batch y0, H0
         params: initial theta0, D0 = inv(H0^T H0), and features used idx0
         predictive_error: initial predictive error
         num_available_features: K
         noise_variance: observation noise var
        '''

        # Initial data
        self.y = initials[0]
        self.H = initials[1]

        # Initial params
        self.theta = params[0]
        self.D = params[1]
        self.selected_features_idx = params[2]
        self.PredError = predictive_error

        # System settings
        self.K = num_available_features
        self.var = noise_variance

        # Copy of Feature matrix that will be resorted (for convenience) and used throughout the algorithm
        self.H_sorted = initials[1]

        # To keep track of the feature resorting
        self.all_features_idx = np.arange(self.K)
        self.sorted_features_idx = np.arange(self.K)

        # Present model size and initial data size
        self.k = len(self.theta)
        self.t0 = len(self.y)



    # MODEL UPDATE WITH NEW DATA (FIXED TIME) --------------------------------------------------------------
    def model_update(self, data_t: float, features_t: np.ndarray, t: int):

        '''
        Function that executes model update in Algorithm 4
        (data_t, features_t) - new data pair at time t
        '''

        # New data point at time t
        self.y = np.append(self.y, data_t)
        self.H = np.vstack((self.H, features_t))

        # Append new feature to resorted feature matrix
        self.H_sorted = np.vstack((self.H_sorted, features_t[self.sorted_features_idx]))

        # Update predictive error J(t-1) --> J(t)
        e = data_t - features_t[self.selected_features_idx] @ self.theta
        J = self.PredError + e ** 2

        # Define present model
        jump = ModelJump(self.theta, self.D, self.y, self.H_sorted, self.K, self.var, J, (self.t0, t))

        # STAY
        theta_stay, idx_stay, J_stay, Dk_stay = jump.stay()

        # JUMP UP 
        if self.k < self.K:
            theta_up, idx_up, J_up, Dk_up = jump.up()
        else:
            J_up = float('inf')
            Dk_up = theta_up = idx_up = 0

        # JUMP DOWN
        if self.k > 1:
            theta_down, idx_down, J_down, Dk_down = jump.down()
        else:
            J_down = float('inf')
            Dk_down = theta_down = idx_down = 0

        # Store
        J_jump = [J_stay, J_up, J_down]
        Dk_jump = [Dk_stay, Dk_up, Dk_down]
        idx_jump = [idx_stay, idx_up, idx_down]
        theta_jump = [theta_stay, theta_up, theta_down]

        # Get quantities with smallest predictive error
        self.theta, self.D, self.PredError, self.all_features_idx = util.get_min(theta_jump, Dk_jump, J_jump, idx_jump)

        # Quantities to Update
        self.k = self.theta.shape[0]
        self.H_sorted = self.H_sorted[:, self.all_features_idx]

        # Find selected features location in original feature matrix
        self.sorted_features_idx = util.get_features(self.H[0, :], self.H_sorted[0, :], self.K, self.K)
        self.selected_features_idx = self.sorted_features_idx[:self.k]



    # TIME UPDATE WITH NEW DATA (FIXED MODEL)-  -------------------------------------------------------------
    def time_update(self, data_t: float,  features_t: np.ndarray):

        '''
        Function that executes time update in Algorithm 4
        (data_t, features_t) - new data pair at time t
        '''

        # Define present model k
        present_model = ls.RLS(self.theta, self.D)

        # Update with new data point
        self.theta, self.D = present_model.ascend(data_t, features_t, self.var)