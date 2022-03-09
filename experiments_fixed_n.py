from householder_qr_factorization import *
from column_orthogonality_check import *
from residual import *
import numpy as np
import matplotlib.pyplot as plt
import time


def experiments_fixed_n_function(seed):

    runtimes_householder = []
    runtimes_explicit_qr_product = []
    runtimes_implicit_qr_product = []
    res_qr_explicit_q = []
    res_qr_implicit_q = []
    orthogonality_of_q_columns = []
    problem_sizes_m = []


    n = 100
    print("Fixed n = ", n)

    for iteration in range(1, 3):

        # two rounds of experiments, where:
        # n = 100,200,...,1000 in the first iteration
        # n = 1500,2000,...,3000 in the second iteration
        if iteration == 1:
            start = 100
            end = 1001
            step = 100
        else:
            start = 1500
            end = 3001
            step = 500

        for m in range(start, end, step):

            print("m = ", m)
            problem_sizes_m.append(m)
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
    householder_runtime_m, = plt.semilogy(problem_sizes_m, runtimes_householder, 'b')
    plt.xlabel('m (with fixed n = 100)')
    plt.ylabel('Runtime (in seconds)')
    plt.savefig('runtime_householder_qr_fixed_n.png', dpi=800)
    plt.clf()

    # Runtimes of the two variants of computing the QR product
    fig, axs = plt.subplots(1, 1)
    plt.figure(1)
    explicit_q_runtime, = plt.semilogy(problem_sizes_m, runtimes_explicit_qr_product, 'r')
    implicit_q_runtime, = plt.semilogy(problem_sizes_m, runtimes_implicit_qr_product, 'b')
    plt.xlabel('m (with fixed n = 100)')
    plt.ylabel('Runtime (in seconds)')
    plt.legend([explicit_q_runtime, implicit_q_runtime], ["Explicit QR", "Implicit QR"])
    plt.savefig('runtime_qr_product_fixed_n.png', dpi=800)
    plt.clf()

    # Residuals of the two variants of computing the QR product
    fig, axs = plt.subplots(1, 1)
    plt.figure(1)
    explicit_q_res, = plt.semilogy(problem_sizes_m, res_qr_explicit_q, 'r')
    implicit_q_res, = plt.semilogy(problem_sizes_m, res_qr_implicit_q, 'b')
    plt.xlabel('m (with fixed n = 100)')
    plt.ylabel('Residual')
    plt.legend([explicit_q_res, implicit_q_res], ["Explicit QR", "Implicit QR"])
    plt.savefig('residuals_fixed_n.png', dpi=800)
    plt.clf()

    # Orthogonality of Q's columns (when Q is formed/stored explicitly)
    fig, axs = plt.subplots(1, 1)
    plt.figure(1)
    orthogonality, = plt.semilogy(problem_sizes_m, orthogonality_of_q_columns, 'b')
    plt.xlabel('m (with fixed n = 100)')
    plt.ylabel('||Q^T Q - I||_1')
    plt.savefig('orthogonality_fixed_n.png', dpi=800)
    plt.clf()