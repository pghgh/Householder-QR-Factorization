from extract_r_from_a import *
import numpy as np


def qr_implicit_q_function(A_overwritten):
    m = A_overwritten.shape[0]-1
    n = A_overwritten.shape[1]
    # in matrix R we will store the QR product (Q is not stored explicitly in this case)
    R = extract_r_from_a_function(A_overwritten)
    for column in range(n-1, -1, -1):
        v = A_overwritten[column+1:m+1, column]
        v.shape = (v.shape[0], 1)
        if np.array_equal(v, np.zeros((v.shape[0], 1))):
            continue
        alpha = 2 / np.dot(v.transpose(), v)
        R[column:m,:] -= alpha * v @ (v.transpose() @ R[column:m,:])
    return R