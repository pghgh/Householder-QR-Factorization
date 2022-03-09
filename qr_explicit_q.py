import numpy as np

def qr_explicit_q_function(A_overwritten):

    m = A_overwritten.shape[0]-1
    n = A_overwritten.shape[1]
    Q = np.eye(m)
    for column in range(n-1, -1, -1):
        v = A_overwritten[column+1:m+1, column]
        v.shape = (v.shape[0], 1)
        if np.array_equal(v, np.zeros((v.shape[0], 1))):
            continue
        alpha = 2 / np.dot(v.transpose(), v)
        Q[column:m, column:m] -= alpha * v @ (v.transpose() @ Q[column:m, column:m])
    return Q