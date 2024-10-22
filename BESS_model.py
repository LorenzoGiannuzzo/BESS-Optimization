import numpy as np

from utils import Get_data
from utils import BESS

from argparser import size, technology
from argparser import soc_min, soc_max, BESS_power, power_energy

# FILE PATH DATA

file_path = r"Input/BESS Data.xlsx"
sheetname = "BESS Properties"
sheetname2 = "Li-ion ChargeDischarge Curve 10"


'''
DEFINE BESS PARAMETERS:

   1) Technology
   2) Nominal Capacity / Nominal Power (Specific Energy / Specific Power)
   3) Size


'''


# BESS Parameters

technology = technology  # Define BESS Technology
se_sp = 10   # Define Nominal Capacity / Nominal Power - Parameter
size = size  # Define BESS size (Nominal Capacity) in kWh

# LOADING FILES

properties = Get_data.get_data(file_path, sheetname)
load_curve = Get_data.get_data(file_path, sheetname2)


# TAKE CHARGE AND DISCHARGE BATTERY LOAD CURVES (to do: still to be automatized)

charge_rate = load_curve.iloc[:356, [0, 3, 5]]
discharge_rate = load_curve.iloc[357:, [0, 4, 5]]


# GET CHOSEN BESS PARAMETERS

BESS_Parameters = BESS.get_bess(technology, properties, se_sp, size)


# INTERPOLATE CHARGE AND DISCHARGE FUNCTIONS FROM CHARGE AND DISCHARGE RATE

charge_rate_interpolated_func, discharge_rate_interpolated_func = BESS.get_c_d_functions(load_curve)


class BESS_model:
    def __init__(self, time_window, PUN_timeseries, soc, size, c_func, d_func, alpha):
        self.time_window = time_window
        self.PUN_timeseries = PUN_timeseries
        self.soc = soc
        self.size = size
        self.c_func = c_func
        self.d_func = d_func
        self.charged_energy = np.zeros(len(PUN_timeseries))
        self.discharged_energy = np.zeros(len(PUN_timeseries))
        self.c_d_timeseries = None
        self.alpha = None

    def run_simulation(self, c_d_timeseries, alpha):
        self.c_d_timeseries = np.array(c_d_timeseries).reshape(self.time_window)
        self.alpha = np.array(alpha)

        for index in range(len(self.PUN_timeseries) - 1):
            if self.c_d_timeseries[index] >= 0.0:
                self.c_d_timeseries[index] = np.minimum(self.c_d_timeseries[index]*np.abs(self.alpha[index]),
                                                        np.minimum(self.c_func(self.soc[index])*np.abs(self.alpha[index]), soc_max - self.soc[index]), np.array(np.array(power_energy)*np.abs(self.alpha[index])))
            else:
                self.c_d_timeseries[index] = np.maximum(self.c_d_timeseries[index]*np.abs(self.alpha[index]),
                                                        np.maximum(-self.d_func(self.soc[index])*np.abs(self.alpha[index]), soc_min - self.soc[index]), np.array(np.array(-power_energy)*np.abs(self.alpha[index])))

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
                self.soc[index + 1] = np.minimum(soc_max, self.soc[index] + self.charged_energy[index] / self.size)
            else:
                self.soc[index + 1] = max(soc_min, self.soc[index] + self.discharged_energy[index] / self.size)

        return self.charged_energy, self.discharged_energy
