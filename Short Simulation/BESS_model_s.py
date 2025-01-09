"""

BESS Optimization using NSGA-III Algorithm

    __author__ = "Lorenzo Giannuzzo"
    __maintainer__ = "Lorenzo Giannuzzo"
    __email__ = "lorenzo.giannuzzo@polito.it"
    __status__ = "in progress"
    __version__ = "v0.2.1"
    __license__ = "MIT"

Last Update of current code: 09/01/2025 - 12:18

"""

# IMPORT LIBRARIES
import numpy as np
from utils import Get_data
from utils import BESS
from argparser_s import size, technology
from argparser_s import soc_min, soc_max, power_energy, n_cycles

# SETTING ST FILE PATHS
file_path = r"Input/BESS Data.xlsx"
sheetname = "BESS Properties"
sheetname2 = "Li-ion ChargeDischarge Curve 10"

# DEFINE BESS PARAMETERS
technology = technology
size = size

#TODO STATIC PARAMETER AT THE MOMENT - IT SHOULD BE A PARAMETER OBTAINABLE FROM THE ARGPARSER
se_sp = 10 #Nominal Capacity / Nominal Power (Specific Energy / Specific Power)

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


# DEFINE DEGRADATION FUNCTION
def degradation(cycle_num):

    capacity_remaining = (
            -0.00000000000000000000000000000005613 * cycle_num ** 9 +
            0.000000000000000000000000003121 * cycle_num ** 8 -
            0.00000000000000000000006353 * cycle_num ** 7 +
            0.000000000000000000663 * cycle_num ** 6 -
            0.000000000000003987 * cycle_num ** 5 +
            0.00000000001435 * cycle_num ** 4 -
            0.0000000307 * cycle_num ** 3 +
            0.00003746 * cycle_num ** 2 -
            0.0277 * cycle_num + 100
    )

    return capacity_remaining

# DEFINE BESS MODEL CLASS
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
        self.n_cycles = n_cycles
        self.soc_max = soc_max

    # DEFINE RUN SIMULATION FUNCTION TO SIMULATE BESS ENERGY FLOWS
    def run_simulation(self, c_d_timeseries):

        # SET CHARGE/DISCHARGE VECTOR (% of SoC charged or discharged at each timestep) WITH SAME LENGTH AS TIME WINDOW
        self.c_d_timeseries = np.array(c_d_timeseries).reshape(self.time_window)

        # EXECUTE THE SIMULATION FOR EACH TIMESTEP
        for index in range(len(self.PUN_timeseries) - 1):

            # EVALUATE BESS MAXIMUM CAPACITY
            max_capacity = degradation(self.n_cycles) / 100
            soc_max = min(self.soc_max, max_capacity)

            # IF BESS IS CHARGING
            if self.c_d_timeseries[index] >= 0.0:

                # EVALUATE THE PHYSICAL CONSTRAINTS OF BESS TO AVOID OVERCHARGNING
                #TODO THIS THING SHOULD BE DONE USING A FUNCTION CONTAINED IN A FILE NAMED CONSTRAINER

                # THE BESS CAN CHARGE THE MINIMUM BETWEEN WHAT'S DEFINED IN C/D_Timeseries (what quantity the agorithm
                # would like to charge), HOW MUCH THE BESS CAN CHARGE BASED ON ITS ACTUAL SoC
                # (self.c_func(self.soc[index])), AND HOW MUCH CAN HE CHARGE LEFT (soc_max - self.soc[index])

                self.c_d_timeseries[index] = np.minimum(
                    self.c_d_timeseries[index],
                    np.minimum(
                        self.c_func(self.soc[index]),
                        soc_max - self.soc[index]
                    )
                )
                self.c_d_timeseries[index] = np.minimum(
                    self.c_d_timeseries[index],
                    np.array(power_energy)
                )

            # IF BESS IS DISCHARGING (specular case as charging)

                # TODO THIS THING SHOULD BE DONE USING A FUNCTION CONTAINED IN A FILE NAMED CONSTRAINER

            else:

                self.c_d_timeseries[index] = np.maximum(
                    self.c_d_timeseries[index],
                    np.maximum(
                        -self.d_func(self.soc[index]),
                        soc_min - self.soc[index]
                    )
                )
                self.c_d_timeseries[index] = np.maximum(
                    self.c_d_timeseries[index],
                    np.array(-power_energy)
                )
            # IF BESS IS DISCHARGING
            if self.c_d_timeseries[index] >= 0:

                # EVALAUTE CHARGED ENERGY INTO THE BESS

                self.charged_energy[index] = self.c_d_timeseries[index] * self.size

                # SET DISCHARGED ENERGY AT 0

                self.discharged_energy[index] = 0.0

            # IF BESS IS DISCHARGING
            elif self.c_d_timeseries[index] < 0:

                # EVALUATE DISCHARGED ENERGY FROM BESS

                self.discharged_energy[index] = self.c_d_timeseries[index] * self.size

                # SET CHARGED ENERGY AT 0

                self.charged_energy[index] = 0.0

            # IF BESS IS NOT CHARGING OR DISCHARGING, SET BOTH C/D ENERGY AT 0
            else:

                self.charged_energy[index] = 0.0

                self.discharged_energy[index] = 0.0

            # UPDATE SoC AT TIMESTEP t + 1

            # IF BESS IS CHARGING
            if self.c_d_timeseries[index] >= 0.0:

                # INCREASE SoC
                self.soc[index + 1] = np.minimum(soc_max, self.soc[index] + self.charged_energy[index] / self.size)

                # RE-EVALUATE CHARGED ENERGY
                self.charged_energy[index] = (self.soc[index+1] - self.soc[index])*self.size
                self.discharged_energy[index] = 0.0

            # IF BESS IS DISCHARGING
            else:

                # DECREASE SoC (discharged_energy is negative)
                self.soc[index + 1] = np.maximum(soc_min, self.soc[index] + self.discharged_energy[index] / self.size)

                # RE-EVALUATE DISCHARGED ENERGY
                self.discharged_energy[index] = (self.soc[index+1] - self.soc[index])*self.size
                self.charged_energy[index] = 0

            # N_CYCLES UPDATE FOR EACH TIMESTEP
            total_energy = self.charged_energy[index] + np.abs(self.discharged_energy[index])
            n_cycles_prev = self.n_cycles
            actual_capacity = size * degradation(n_cycles_prev)
            self.n_cycles = n_cycles_prev + total_energy/actual_capacity

        # RETURN CHARGED AND DISCHARGE ENERGY
        return self.charged_energy, self.discharged_energy
