import numpy as np

def extract_r_from_a_function(A_overwritten):
    # A_overwritten has an additional row at the end
    # so that it could store the Householder vectors
    m = A_overwritten.shape[0]-1
    n = A_overwritten.shape[1]
    R = np.zeros((m, n))
    for column in range(0, n):
        # R is stored in the upper triangle of matrix A_overwritten
        R[0:column+1, column] = A_overwritten[0:column+1, column]
    return R