from experiments_rank_deficient_matrices import *
from experiments_fixed_m import *
from experiments_fixed_n import *

if __name__ == '__main__':
    seed = 123456789 # random seed, which can be modified as needed
    experiments_fixed_n_function(seed)
    experiments_fixed_m_function(seed+1)
    # fixed input matrix dimensions for the experiments
    # with rank deficient matrices; m and n can be modified as needed
    n = 100
    m = 500
    experiments_rank_deficient_matrices_function(seed+2, m, n)
