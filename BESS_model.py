"""

BESS Optimization using NSGA-III Algorithm

    __author__ = "Lorenzo Giannuzzo"
    __maintainer__ = "Lorenzo Giannuzzo"
    __email__ = "lorenzo.giannuzzo@polito.it"
    __status__ = "in progress"
    __version__ = "v0.2.1"
    __license__ = "MIT"

Last Update of current code: 06/11/2024 - 16:13

"""

# IMPORT LIBRARIES

import numpy as np
from utils import Get_data
from utils import BESS
from argparser import size, technology
from argparser import soc_min, soc_max, BESS_power, power_energy

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

# DEFINE BESS MODEL CLASS

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

    # DEFINE RUN SIMULATION FUNCTION TO SIMULATE BESS ENERGY FLOWS

    def run_simulation(self, c_d_timeseries, alpha):

        # SET CHARGE/DISCHARGE VECTOR (% of SoC charged or discharged at each timestep) WITH SAME LENGTH AS TIME WINDOW

        self.c_d_timeseries = np.array(c_d_timeseries).reshape(self.time_window)

        # SET ALPHA VALUES (parameter used to slow down charge/discharge of the BESS)
        #TODO - this parameter is used to take into account BESS degradation in the objective function, we should
        # consider a normal polynomial function instead of this

        self.alpha = np.array(alpha)

        # EXECUTE THE SIMULATION FOR EACH TIMESTEP

        for index in range(len(self.PUN_timeseries) - 1):

            # IF BESS IS CHARGING

            if self.c_d_timeseries[index] >= 0.0:

                # EVALUATE THE PHYSICAL CONSTRAINTS OF BESS TO AVOID OVERCHARGNING
                #TODO THIS THING SHOULD BE DONE USING A FUNCTION CONTAINED IN A FILE NAMED CONSTRAINER

                # THE BESS CAN CHARGE THE MINIMUM BETWEEN WHAT'S DEFINED IN C/D_Timeseries (what quantity the agorithm
                # would like to charge), HOW MUCH THE BESS CAN CHARGE BASED ON ITS ACTUAL SoC
                # (self.c_func(self.soc[index])), AND HOW MUCH CAN HE CHARGE LEFT (soc_max - self.soc[index])

                self.c_d_timeseries[index] = np.minimum(
                    self.c_d_timeseries[index] * np.abs(self.alpha[index]),
                    np.minimum(
                        self.c_func(self.soc[index]) * np.abs(self.alpha[index]),
                        soc_max - self.soc[index]
                    )
                )
                self.c_d_timeseries[index] = np.minimum(
                    self.c_d_timeseries[index],
                    np.array(power_energy) * np.abs(self.alpha[index])
                )
            # IF BESS IS DISCHARGING (specular case as charging)

                # TODO THIS THING SHOULD BE DONE USING A FUNCTION CONTAINED IN A FILE NAMED CONSTRAINER

            else:

                self.c_d_timeseries[index] = np.maximum(
                    self.c_d_timeseries[index] * np.abs(self.alpha[index]),
                    np.maximum(
                        -self.d_func(self.soc[index]) * np.abs(self.alpha[index]),
                        soc_min - self.soc[index]
                    )
                )
                self.c_d_timeseries[index] = np.maximum(
                    self.c_d_timeseries[index],
                    np.array(-power_energy) * np.abs(self.alpha[index])
                )
            # IF BESS IS DISCHARGING

            if self.c_d_timeseries[index] >= 0:

                # EVALAUTE CHARGED ENERGY INTO THE BESS

                self.charged_energy[index] = self.c_d_timeseries[index] * self.size

                # SET DISCHARGED ENERGY AT 0

                self.discharged_energy[index] = 0.0

            # IF BESS IS DISCHARGING

            elif self.c_d_timeseries[index] <= 0:

                # EVALUATE DISCHARGED ENERGY FROM BESS

                self.discharged_energy[index] = self.c_d_timeseries[index] * self.size

                # SET CHARGED ENERGY AT 0

                self.charged_energy[index] = 0.0

            # IF BESS IS NOT CHARGING OR DISCHARGING, SET BOTH C/D ENERGY AT 0

            else:

                self.charged_energy[index] = 0.0

                self.discharged_energy[index] = 0.0

            # UPDATE SoC AT TIMESTEP t + 1
            #TODO the update of SoC is useless now

            # IF BESS IS CHARGING

            if self.c_d_timeseries[index] >= 0.0:

                # INCREASE SoC

                self.soc[index + 1] = np.minimum(soc_max, self.soc[index] + self.charged_energy[index] / self.size)

            # IF BESS IS DISCHARGING

            else:

                # DECREASE SoC (discharged_energy is negative)

                self.soc[index + 1] = max(soc_min, self.soc[index] + self.discharged_energy[index] / self.size)


        # RETURN CHARGED AND DISCHARGE ENERGY

        return self.charged_energy, self.discharged_energy
