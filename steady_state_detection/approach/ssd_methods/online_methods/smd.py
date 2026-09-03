# Code source: https://github.com/cepalium/online-steady-state-detection/tree/master/ossdem
# Robert Bethea, Applied Engineering Statistics, Linear Regression, vol. 121. Marcel Dekker, 1991.

import numpy as np
from sklearn.linear_model import LinearRegression

class SlopeDetectionMethod:
    def __init__(self, slope_crit):
        self.name = 'sdm'
        self.data = list()  # collection of batch data
        self.i = 0  # index at current unchecked batch
        self.slope_crit = slope_crit  # critical slope value: if estimated slope < critical slope, batch is in steady-state
        self.size = 0  # no. inserted batches
        self.ss_start_point = -1  # detected steady-state starting point
        self.slope = list()  # list: slope at batch i

    def insert(self, batch):
        """ add & store new batch of data """
        self.data.append(batch)
        self.size += 1

    def steady_state_start_point(self):
        """ return earliest detected steady-state starting point """
        self.update()
        if self.ss_start_point != -1:
            return self.ss_start_point
        # find the earliest detected steady-state starting point
        n = len(self.data[0]) if self.size > 0 else 0
        for i in range(self.size):  # check until the last batch
            if abs(self.slope[i]) < self.slope_crit:
                self.ss_start_point = n * i
                break
        return self.ss_start_point

    def update(self):
        """ check state for every unchecked batch """
        while self.i < self.size:
            model = LinearRegression()
            t = np.arange(len(self.data[self.i]))
            model.fit(t[:, np.newaxis], self.data[self.i])
            slope_deg = np.arctan2(model.coef_[0], 1)  # slope in degree
            self.slope.append(slope_deg)
            self.i += 1  # move to next batch
        return


def conduct_ssd_sdm(signal, batch_size_min: int = 10, slope_crit: float = 0.002):

    signal_length = len(signal)
    batch_size = int(max(0.01*signal_length, batch_size_min))
    batches = [signal.values[i * batch_size: (i + 1) * batch_size] for i in range((len(signal) + batch_size - 1) // batch_size)]
    slope_detector = SlopeDetectionMethod(slope_crit=slope_crit)
    print("Batch size = {}, slope_crit = {}".format(batch_size, slope_crit))
    #
    for i, batch in enumerate(batches):
        slope_detector.insert(batch)
        T_hat = slope_detector.steady_state_start_point()
        print("Batch {} ({}-{}) - Detected steady state start point = {}".format(i, i*batch_size, (i+1)*batch_size, T_hat))
