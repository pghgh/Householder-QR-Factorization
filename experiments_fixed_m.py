from householder_qr_factorization import *
from column_orthogonality_check import *
from residual import *
import numpy as np
import matplotlib.pyplot as plt
import time

def experiments_fixed_m_function(seed):

    runtimes_householder = []
    runtimes_explicit_qr_product = []
    runtimes_implicit_qr_product = []
    res_qr_explicit_q = []
    res_qr_implicit_q = []
    orthogonality_of_q_columns = []
    problem_sizes_n = []

    m = 3000
    print("Fixed m = ", m)


    for iteration in range(1, 2):

        # one round of experiments, where n = 100,200,...,500
        if iteration == 1:
            start = 100
            end = 501
            step = 100

        for n in range(start, end, step):
            print("n = ", n)
            problem_sizes_n.append(n)
            np.random.seed(seed)
            A = np.random.rand(m, n)
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
            runtimes_implicit_qr_product.append(elapsed_time_implicit_qr_product)
            orthogonality_of_q_columns.append(column_orthogonality_check_function(A_overwritten))

    # Plotting the data
    plt.rcParams.update({'font.size': 6})
    # Runtimes of Householder QR Factorization
    fig, axs = plt.subplots(1, 1)
    plt.figure(1)
    householder_runtime_n, = plt.semilogy(problem_sizes_n, runtimes_householder, 'b')
    plt.xlabel('n (with fixed m = 3000)')
    plt.ylabel('Runtime (in seconds)')
    plt.savefig('runtime_householder_qr_fixed_m.png', dpi=800)
    plt.clf()

    # Runtimes of the two variants of computing the QR product
    fig, axs = plt.subplots(1, 1)
    plt.figure(1)
    explicit_q_runtime, = plt.semilogy(problem_sizes_n, runtimes_explicit_qr_product, 'r')
    implicit_q_runtime, = plt.semilogy(problem_sizes_n, runtimes_implicit_qr_product, 'b')
    plt.xlabel('n (with fixed m = 3000)')
    plt.ylabel('Runtime (in seconds)')
    plt.legend([explicit_q_runtime, implicit_q_runtime], ["Explicit QR", "Implicit QR"])
    plt.savefig('runtime_qr_product_fixed_m.png', dpi=800)
    plt.clf()

    # Residuals of the two variants of computing the QR product
    fig, axs = plt.subplots(1, 1)
    plt.figure(1)
    explicit_q_res, = plt.semilogy(problem_sizes_n, res_qr_explicit_q, 'r')
    implicit_q_res, = plt.semilogy(problem_sizes_n, res_qr_implicit_q, 'b')
    plt.xlabel('n (with fixed m = 3000)')
    plt.ylabel('Residual')
    plt.legend([explicit_q_res, implicit_q_res], ["Explicit QR", "Implicit QR"])
    plt.savefig('residuals_fixed_m.png', dpi=800)
    plt.clf()

    # Orthogonality of Q's columns (when Q is formed/stored explicitly)
    fig, axs = plt.subplots(1, 1)
    plt.figure(1)
    orthogonality, = plt.semilogy(problem_sizes_n, orthogonality_of_q_columns, 'b')
    plt.xlabel('n (with fixed m = 3000)')
    plt.ylabel('||Q^T Q - I||_1')
    plt.savefig('orthogonality_fixed_m.png', dpi=800)
    plt.clf()