# Source: https://github.com/auralius/steady-state-detection/blob/main/kelly/ssd.m
# R. R. Rhinehart, Automated steady and transient state identification in
# noisy processes, in Proceedings of the American Control Conference, 2013,
# no. June 2013, pp. 4477–4493.
# Section V

import numpy as np

def ssd(x, n, t_crit):
    P = np.zeros(len(x))

    if n > len(x):
        n = len(x)

    k = 0
    should_break = False

    while True:
        start = k
        end = start + n

        if end > len(x):
            end = len(x)
            n = end - start
            should_break = True

        x_active = x[start:end]

        # Estimate the slope m of the drift component using first differencing
        m = 0
        for t in range(1, n):
            m += (x_active[t] - x_active[t - 1])
        m /= n

        # Calculate mu (mean of the adjusted values)
        mu = (sum(x_active) - sum(range(1, n + 1)) * m) / n

        # Calculate the standard deviation sd
        sd = 0
        for t in range(n):
            sd += (x_active[t] - m * (t + 1) - mu) ** 2
        sd = np.sqrt(sd / (n - 2))

        # Calculate the proportion y of points within the t_crit threshold
        y = 0
        for t in range(n):
            if abs(x_active[t] - mu) <= t_crit * sd:
                y += 1
        y /= n

        P[start:end] = y

        if should_break:
            break

        k += n

    return P
