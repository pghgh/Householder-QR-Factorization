from qr_product import *
import scipy as sp

def residual_function(A, A_overwritten, q_storage):
    return sp.linalg.norm(A - qr_product_function(A_overwritten, q_storage), 1) / sp.linalg.norm(A, 1)