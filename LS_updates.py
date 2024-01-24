import numpy as np

# ORDER RECURSIVE LEAST SQUARES ==========================================================
class ORLS:

    ''' Updates theta from dimension k to
     ascend:  k --> k + 1
     descend: k --> k - 1
     '''

    def __init__(self, theta: np.ndarray, D: np.ndarray, y: np.ndarray, H: np.ndarray, K: int):
        '''
        Args:
            theta: parameter to be updated of size k x 1
            D: Inverse feature matrix to be updated
            y: Output data of size t x 1
            H: Feature matrix of size t x K
            K: Number of total available features
        '''

        self.theta = theta
        self.D = D
        self.k = len(theta)
        self.K = K
        self.y = y
        self.H = H


    # ASCENDING STEP -----------------------------------------------------------------------
    def ascend(self, m: int):
        '''
        Args:
            m: Index of Feature to be added
        '''

        # Time instant and parameter dimension
        t = len(self.y)
        k = self.k

        # Current input data (loops 0 to index minus 1)
        Hk = self.H[:t, :k]

        # New feature to be added
        h_new = self.H[:t, k + m]

        # Projection matrix
        DHkT = self.D @ Hk.T
        P_norm = np.eye(t) - Hk @ DHkT

        # Some reusable terms
        v = h_new @ P_norm
        vy = v @ self.y[:t]
        d = DHkT @ h_new

        # Compute terms of D(k+1)
        D22 = 1 / (v @ h_new)
        D12 = - d * D22
        D11 = self.D + np.outer(d, d) * D22

        # Create D(k+1)
        top = np.vstack([D11, D12])
        D = np.hstack([top, np.append(D12.T, D22).reshape(k + 1, 1)])

        # Update theta(k) to theta(k+1)

        vyD = vy * D22  # reusable term
        theta = np.vstack([self.theta - vyD * d.reshape(k, 1), vyD])

        # New index order
        # Reorder available features (put new feature m right after the other used features)
        idx_H = np.concatenate((np.arange(k), [k + m], np.setdiff1d(np.arange(k, self.K), [k + m])))

        # Update Hk in time for input to Predictive Error
        Hk = self.H[:t, idx_H[:k + 1]]

        return theta, D, idx_H, Hk


    # DESCENDING STEP --------------------------------------------------------------------------
    def descend(self, m: int):
        '''
        Args:
            m: Index of Feature to be removed
        '''

        # System Dimension
        k = self.k
        t = len(self.y)

        # Range of indices to include
        idx = np.setdiff1d(np.arange(k), m)
        idx_m = np.append(idx, m)

        # Get D(k+1) bar
        Dswap = np.empty((k, k))
        Dswap[:k - 1, :][:, :k - 1] = self.D[idx, :][:, idx]
        Dswap[k - 1, :k - 1] = self.D[m, idx]
        Dswap[:, k - 1] = self.D[idx_m, m]

        # Get D(k+1) bar blocks
        D11 = Dswap[:k - 1, :][:, :k - 1]
        D12 = Dswap[:k - 1, k - 1:]
        D22 = Dswap[k - 1, k - 1]

        # Get D(k) final
        D = D11 - np.outer(D12, D12.T) / D22

        # Update rest of theta
        theta = self.theta[idx] - self.theta[m] * (D12 / D22).reshape(k - 1, 1)

        # New index order
        # Hk to input to Predictive Error
        Hk = self.H[:, idx_m]

        # Reorder available features - move m to last
        idx_H = np.concatenate((idx, np.arange(k, self.K), [m]))

        return theta, D, idx_H, Hk


class RLS:

    def __init__(self, theta, D):
        '''
        Args:
            theta: parameter to be updated
            D: Inverse feature matrix to be updated
            k: Dimension of current parameter theta
        '''

        self.theta = theta
        self.D = D
        self.k = len(theta)

    def ascend(self, y_n: float, h_n: np.ndarray, var_y: float):
        '''
        Args:
         y_n: nth data point 
         h_n: the nth data vector of features 
         var_y: the variance of the model noise 
        '''

        # Current error
        e = y_n - h_n @ self.theta

        # Find current sigma
        Sigma = var_y * self.D

        # Update gain
        temp = Sigma @ h_n
        K = temp / (var_y + h_n @ temp)

        # Update estimate
        self.theta = self.theta + e * K.reshape(self.k, 1)

        # Update covariance
        temp = np.eye(self.k) - np.outer(K, h_n)
        Sigma = np.matmul(temp, Sigma)

        # Update D
        self.D = Sigma / var_y

        return self.theta, self.D



class PredictiveError:

    '''
    compute:
    Computes the change in predictive error from some initial t0 up to some t when switching model dimension:
    k --> k + 1
    k --> k - 1
    Starting from some inital t0 point up to some end point t
    '''

    def __init__(self, y, t0, t, K, var_y):

        '''
            t0: Starting data point
            t: Ending data point (total # of data used t-t0)
            K: Number of total available features
            var_y: Noise variance
            y: All available data
        '''

        self.t0 = t0
        self.t = t
        self.K = K
        self.var_y = var_y
        self.y = y[:t + 1]

    # PREDICTIVE ERROR 
    def compute(self, Hk, k):

        '''
        Hk: Feature matrix up to (including) time t and features k+1
        k: Dimension of present model

        Note that:
         if computing a model down, input Hk including the feature needed to be removed in the last column
        and k as (k-1).
        If computing a model up, input Hk including the feature needed to be added in the last column, and k as k

        '''

        # Time start and end
        t0 = self.t0
        t = self.t

        # START k x k
        Dk = np.linalg.inv(Hk[:t0, :k].T @ Hk[:t0, :k])
        theta_k = Dk @ Hk[:t0, :k].T @ self.y[:t0]
        theta_k = theta_k.reshape(len(theta_k), 1)

        # Update to k+1 x k+1 ..............................
        model_up = ORLS(theta_k, Dk, self.y[:t0], Hk, self.K)
        theta_kk, Dkk, _, _ = model_up.ascend(0)

        # Initialize
        G = 0
        THETA = theta_k

        # Create time instance at t0 to RLS update up to t
        model_k = RLS(theta_k, Dk)
        model_kk = RLS(theta_kk, Dkk)

        # Time increments
        for i in range(t0, t):

            # Compute Qi
            Q = Hk[i, :k] @ (-Dkk[:k, k] / Dkk[k, k]) - Hk[i, k]

            # Compute Gi
            G = np.hstack([G, Q * theta_kk[-1]])

            # Store THETAs using t0 elements
            THETA = np.hstack([THETA, theta_k])

            # Update thetas
            theta_k, Dk = model_k.ascend(self.y[i], Hk[i, :k], self.var_y)
            theta_kk, Dkk = model_kk.ascend(self.y[i], Hk[i, :k + 1], self.var_y)

        # Residual error (ignore initialized element in THETA)
        temp = Hk[t0:t, :k] * THETA[:, :][:, 1:].T
        E = (self.y[t0:t] - np.sum(temp, axis=1)).reshape(t-t0, 1)

        # Ignore initialized element in G, and reshape
        G = G[1:].reshape(t-t0, 1)

        return G, E