# Code source: https://github.com/cepalium/online-steady-state-detection/tree/master/ossdem
# Martin Boesler, “Steady state detection for computational fluid dynamics,” 2017.



import numpy as np


class FTest:
    def __init__(self, F_crit):
        self.name = 'f_test'
        self.data = list()  # collection of data in batch
        self.i = 0  # index at current unchecked batch
        self.size = 0  # no. received batches
        self.F_crit = F_crit  # critical F value: if estimated F < critical F, batch is steady-state
        self.F = list()  # list: F-value at batch i
        self.ss_start_point = -1  # detected steady-state starting point

    def insert(self, batch):
        self.data.append(batch)
        self.size += 1

    def steady_state_start_point(self):
        """ return earliest detected steady-state starting point """
        self.update()
        if self.ss_start_point != -1:
            return self.ss_start_point
        # find the earliest steady-state starting point
        n = len(self.data[0]) if self.F else 0
        for i in range(len(self.F)):  # check until second last batch
            if self.F[i] < self.F_crit:
                self.ss_start_point = n * i
                break
        return self.ss_start_point

    def update(self):
        """ update state of second last batch """
        if self.size < 2:  # trivial case: 0 batch received
            return
        while self.i < self.size - 1:  # check to 2nd last batch
            s1 = np.var(self.data[self.i])  # var_s_i
            s2 = np.var(self.data[self.i + 1])  # var_s_i+1
            F_i = s1 / s2 if s1 > s2 else s2 / s1  # F-value
            self.F.append(F_i)
            self.i += 1  # move to next batch
        return


def conduct_ssd_f_test(signal, batch_size_min: int = 10, F_crit: float = 1.2):

    signal_length = len(signal)
    batch_size = int(max(0.01*signal_length, batch_size_min))
    batches = [signal.values[i * batch_size: (i + 1) * batch_size] for i in range((len(signal) + batch_size - 1) // batch_size)]
    f_test = FTest(F_crit=F_crit)
    print("Batch size = {}, F_crit = {}".format(batch_size, F_crit))
    for i, batch in enumerate(batches):
        f_test.insert(batch)
        T_hat = f_test.steady_state_start_point()
        print(
            "Batch {} ({}-{}) - Detected steady state start point = {}".format(i, i * batch_size, (i + 1) * batch_size,
                                                                               T_hat))
