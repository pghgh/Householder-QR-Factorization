from qr_explicit_q import *
import numpy as np
import scipy as sp

def column_orthogonality_check_function(A_overwritten):
    Q_explicit = qr_explicit_q_function(A_overwritten)
    return sp.linalg.norm(Q_explicit.transpose() @ Q_explicit - np.eye(Q_explicit.shape[0]), 1)