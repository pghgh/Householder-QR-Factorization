from qr_implicit_q import *
from qr_explicit_q import *
from extract_r_from_a import *

def qr_product_function(A_overwritten, q_storage):
    if q_storage == "explicit": # explicit storage of Q is wished
        return qr_explicit_q_function(A_overwritten) @ extract_r_from_a_function(A_overwritten) # multiply Q and R, where both of them are stored explicitly
    return qr_implicit_q_function(A_overwritten) # otherwise we multiply Q with R without explicitly storing Q