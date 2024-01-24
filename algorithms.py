import numpy as np


class ORLS:

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

    # ASCENDING STEP ====================================
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

        # New intex order
        # Reorder available features (put new feature m right after the other used features)
        idx_H = np.concatenate((np.arange(k), [k + m], np.setdiff1d(np.arange(k, self.K), [k + m])))

        # Update Hk in time for input to Predictive Error
        Hk = self.H[:t, idx_H[:k + 1]]

        return theta, D, idx_H, Hk

    # DESCENDING STEP ====================================
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