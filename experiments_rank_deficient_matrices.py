import numpy as np
import matplotlib.pyplot as plt
from householder_qr_factorization import *
from residual import *
from column_orthogonality_check import *
import time


def experiments_rank_deficient_matrices_function(seed, m, n):
    np.random.seed(seed)
    A = np.random.rand(m, n)
    random_column = np.random.randint(0, n)
    factor = 0.1
    runtimes_householder = []
    runtimes_explicit_qr_product = []
    runtimes_implicit_qr_product = []
    res_qr_explicit_q = []
    res_qr_implicit_q = []
    orthogonality_of_q_columns = []
    condition_numbers = []

    for iteration in range(0, 10):
        A[:, random_column] *= factor
        cond_number = np.linalg.cond(A, 2)
        print("Condition number of input matrix = ", cond_number)
        condition_numbers.append(cond_number)
        A_overwritten, elapsed_time = householder_qr_factorization_function(A)
        runtimes_householder.append(elapsed_time)
        start = time.time()
        res_qr_explicit_q.append(residual_function(A, A_overwritten, "explicit"))
        end = time.time()
        elapsed_time_explicit_qr_product = end - start
        runtimes_explicit_qr_product.append(elapsed_time_explicit_qr_product)
        start = time.time()
        res_qr_implicit_q.append(residual_function(A, A_overwritten, "implicit"))
        end = time.time()
        elapsed_time_implicit_qr_product = end - start
        runtimes_implicit_qr_product.append(elapsed_time_explicit_qr_product)
        orthogonality_of_q_columns.append(column_orthogonality_check_function(A_overwritten))

    # Plotting the data
    plt.rcParams.update({'font.size': 6})
    # Runtimes of Householder QR Factorization
    fig, axs = plt.subplots(1, 1)
    plt.figure(1)
    householder_runtime_m, = plt.semilogy(condition_numbers, runtimes_householder, 'b')
    x_label = "Condition numbers of input matrix (fixed m = "  + str(m) + ", n = " + str(n) + ")"
    plt.xlabel(x_label)
    plt.ylabel('Runtime (in seconds)')
    plt.savefig('runtime_rank_deficient_matrix.png', dpi=800)
    plt.clf()

    # Runtimes of the two variants of computing the QR product
    fig, axs = plt.subplots(1, 1)
    plt.figure(1)
    explicit_q_runtime, = plt.semilogy(condition_numbers, runtimes_explicit_qr_product, 'r')
    implicit_q_runtime, = plt.semilogy(condition_numbers, runtimes_implicit_qr_product, 'b')
    plt.xlabel(x_label)
    plt.ylabel('Runtime (in seconds)')
    plt.legend([explicit_q_runtime, implicit_q_runtime], ["Explicit QR", "Implicit QR"])
    plt.savefig('runtime_qr_product_rank_deficient_matrix.png', dpi=800)
    plt.clf()

    # Residuals of the two variants of computing the QR product
    fig, axs = plt.subplots(1, 1)
    plt.figure(1)
    explicit_q_res, = plt.semilogy(condition_numbers, res_qr_explicit_q, 'r')
    implicit_q_res, = plt.semilogy(condition_numbers, res_qr_implicit_q, 'b')
    plt.xlabel(x_label)
    plt.ylabel('Residual')
    plt.legend([explicit_q_res, implicit_q_res], ["Explicit QR", "Implicit QR"])
    plt.savefig('residuals_rank_deficient_matrix.png', dpi=800)
    plt.clf()

    # Orthogonality of Q's columns
    fig, axs = plt.subplots(1, 1)
    plt.figure(1)
    orthogonality, = plt.semilogy(condition_numbers, orthogonality_of_q_columns, 'b')
    plt.xlabel(x_label)
    plt.ylabel('||Q^T Q - I||_1')
    plt.savefig('orthogonality_rank_deficient_matrix.png', dpi=800)
    plt.clf()