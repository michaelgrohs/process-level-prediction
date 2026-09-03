# Source: https://github.com/auralius/steady-state-detection/blob/main/rhinehart/CalculateR.m
# R.R.Rhinehart, Automated steady and transient state identification in noisy processes,
# in Proceedings of the American Control (Conference, 2013, no.June) 2013, pp. 4477–4493.
# Section II

import numpy as np

def calculate_r(x, l1, l2, l3):
    N = len(x)

    cl1 = 1 - l1
    cl2 = 1 - l2
    cl3 = 1 - l3

    xf = 0
    nu2f = 0
    delta2f = 0
    measurement_old = 0

    R = np.zeros(N)

    for i in range(N):
        measurement = x[i]
        nu2f = l2 * (measurement - xf) ** 2 + cl2 * nu2f
        xf = l1 * measurement + cl1 * xf
        delta2f = l3 * (measurement - measurement_old) ** 2 + cl3 * delta2f
        measurement_old = measurement

        R[i] = (2 - l1) * nu2f / delta2f

    return R

#  Section V
# Source: https://github.com/auralius/steady-state-detection/blob/main/rhinehart/CalculateR_N.m
def calculate_r_n(x, N):
    R = np.zeros(len(x))

    if N > len(x):
        N = len(x)

    k = 0

    while True:
        start = k
        end = start + N

        if end > len(x):
            break

        sum1 = 0
        sum2 = 0
        sum3 = 0

        for i in range(start, end):
            sum1 += x[i] ** 2
            sum2 += x[i]
            sum3 += (x[i + 1] - x[i]) ** 2

        R[start:end] = 0.5 * (sum1 - sum2 ** 2 / N) / sum3  # Eq. 15

        k += N

    return R