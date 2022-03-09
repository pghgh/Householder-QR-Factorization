# Householder-QR-Factorization

This repository contains the Householder QR Factorization of randomly generated matrices, often used for solving Linear Least Squares problems, which was implemented for the course [Numerical High Performance Algorithms, University of Vienna, Winter Semester 2021](https://ufind.univie.ac.at/de/course.html?lv=052112&semester=2021W). Various aspects were analyzed in this context, such as the efficiency of the implementation and the accuracy of the results, the differences between computing the QR product implicitly or explicitly, and how do ill-conditioned matrices influence (or not) the matrix decomposition.

## Running experiments

By running the file *experiments.py*, one can see the outputs of various experiments. The user can also change some variable values, where indicated in the comments (e.g. the seed value for the random numbers generator).

