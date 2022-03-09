import time
import numpy as np
import scipy as sp
from scipy import linalg

def householder_qr_factorization_function(A):

    start = time.time()
    print("Start of method: householder_qr_factorization_function")

    m = A.shape[0]
    n = A.shape[1]
    A = np.concatenate((A, np.zeros(n).reshape((1, n))), axis=0)

    for column in range(0, n):
        a = A[column:m,column]
        a.shape = (a.shape[0], 1)
        e = np.zeros(m-column)
        e[0] = 1.0
        e.shape = (e.shape[0], 1)
        a_norm = sp.linalg.norm(a,2)
        if a[0]>0:
            a_norm = -a_norm
        v = a - a_norm * e
        alpha = 2 / np.dot(v.transpose(), v)
        A[column:m, column:n] -= alpha * v @ (v.transpose() @ A[column:m, column:n])
        A[column+1:m+1, column:column+1] = v

    end = time.time()
    elapsed_time = end - start
    return A, elapsed_time