import numpy as np
def compute_Ep(y, H, p, idx, t, var_y):


    diff = []
    for i in range(p):


        # Features of p-1
        idx0 = np.setdiff1d(idx, idx[i])

        # Resort p features
        fidx = list(idx0) + [idx[i]]

        Dpp = np.linalg.inv(H[:t - 1, fidx].T @ H[:t - 1, fidx])

        # Compute left
        left = H[:t, idx[i]].T @ H[:t, idx[i]]
        R1a = Dpp[:p-1, p-1] @ H[:t, idx0].T / Dpp[p-1,p-1]
        R1 = R1a @ R1a.T
        R2 = 1 / (var_y * Dpp[p-1,p-1])
        right = R1 - R2

        # Difference
        diff.append(left - right)

    return np.sign(diff), diff
