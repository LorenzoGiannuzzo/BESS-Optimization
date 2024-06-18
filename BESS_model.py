import numpy as np

class BESS_model:
    def __init__(self, time_window, PUN_timeseries, soc, size, c_func, d_func):
        self.time_window = time_window
        self.PUN_timeseries = PUN_timeseries
        self.soc = soc
        self.size = size
        self.c_func = c_func
        self.d_func = d_func
        self.charged_energy = np.zeros(len(PUN_timeseries))
        self.discharged_energy = np.zeros(len(PUN_timeseries))
        self.c_d_timeseries = None

    def run_simulation(self, c_d_timeseries):
        self.c_d_timeseries = np.array(c_d_timeseries).reshape(self.time_window)

        for index in range(len(self.PUN_timeseries) - 1):
            if self.c_d_timeseries[index] >= 0.0:
                self.c_d_timeseries[index] = np.minimum(self.c_d_timeseries[index],
                                                        np.minimum(self.c_func(self.soc[index]), 0.9 - self.soc[index]))
            else:
                self.c_d_timeseries[index] = np.maximum(self.c_d_timeseries[index],
                                                        np.maximum(-self.d_func(self.soc[index]), 0.1 - self.soc[index]))

            if self.c_d_timeseries[index] >= 0:
                self.charged_energy[index] = self.c_d_timeseries[index] * self.size
                self.discharged_energy[index] = 0.0
            elif self.c_d_timeseries[index] <= 0:
                self.discharged_energy[index] = self.c_d_timeseries[index] * self.size
                self.charged_energy[index] = 0.0
            else:
                self.charged_energy[index] = 0.0
                self.discharged_energy[index] = 0.0

            if self.c_d_timeseries[index] >= 0.0:
                self.soc[index + 1] = np.minimum(1, self.soc[index] + self.charged_energy[index] / self.size)
            else:
                self.soc[index + 1] = max(0.0, self.soc[index] + self.discharged_energy[index] / self.size)

        return self.charged_energy, self.discharged_energy