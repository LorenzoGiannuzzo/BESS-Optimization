import matplotlib
import numpy as np
from pymoo.core.problem import Problem
from pymoo.core.problem import ElementwiseProblem

import ExcelOpener
import Interpolator

# FILE PATH DATA
file_path = "BESS Data.xlsx"
file_path2 = "PUN.xlsx"
sheetname = "BESS Properties"
sheetname2 = "Li-ion ChargeDischarge Curve 10"
sheetname3 = "PUN"


class Get_data:
    @staticmethod
    def get_data(file_path2, sheetname3):
        data = ExcelOpener.import_file.load_excel(file_path2, sheetname3)
        return data


class BESS:

    @staticmethod
    def get_bess(technology, se_sp, size):
        BESS_Parameters = Properties.iloc[:, [0, 1, 7, 9, 14, 15]]
        BESS_Parameters = BESS_Parameters[BESS_Parameters['Technology'] == technology]
        BESS_Parameters = BESS_Parameters[BESS_Parameters['Specific Energy / Specific Power'] == se_sp]
        BESS_Parameters['Size [kWh]'] = size
        BESS_Parameters = BESS_Parameters.iloc[:, [0, 1, 2, 3, 6]]

        return BESS_Parameters

    @staticmethod
    def get_c_d_functions(load_curve):
        # Select charge_rate and discharge_rate DataFrames from load_curve
        charge_rate = load_curve.iloc[:356, [0, 3.5]]
        discharge_rate = load_curve.iloc[357:, [0, 4, 5]]
        # Interpolate data for charge_rate and discharge_rate
        charge_interpolator = Interpolator.DataInterpolator(charge_rate, 'SoC [%]', 'Charge Rate [kWh/(kWhp*h)]')
        charge_rate_interpolated_func = charge_interpolator.interpolate()
        discharge_interpolator = Interpolator.DataInterpolator(discharge_rate, 'SoC [%]',
                                                               'Discharge Rate [kWh/(kWhp*h)]')
        discharge_rate_interpolated_func = discharge_interpolator.interpolate()

        return charge_rate_interpolated_func, discharge_rate_interpolated_func


# Set matplotlib backend to 'Agg' for non-interactive plotting
matplotlib.use('Agg')

# FILE PATH DATA
file_path = "BESS Data.xlsx"
file_path2 = "PUN.xlsx"
sheetname = "BESS Properties"
sheetname2 = "Li-ion ChargeDischarge Curve 10"
sheetname3 = "PUN"

# BESS DATA
technology = "Li-ion"
se_sp = 10  # Specific Energy / Specific Power
size = 2500  # kWh

# LOADING FILES
Properties = Get_data.get_data(file_path, sheetname)
PUN = Get_data.get_data(file_path2, sheetname3)
load_curve = Get_data.get_data(file_path, sheetname2)

# Select charge_rate and discharge_rate DataFrames from load_curve
charge_rate = load_curve.iloc[:356, [0, 3, 5]]
discharge_rate = load_curve.iloc[357:, [0, 4, 5]]

# GET BESS PARAMETERS
BESS_Parameters = BESS.get_bess(technology, se_sp, size)

# GET C_D(SoC) FUNCTIONS
charge_rate_interpolated_func, discharge_rate_interpolated_func = BESS.get_c_d_functions(load_curve)

time_window = 48


class Revenues(ElementwiseProblem):
    def __init__(
            self,
            size,
            pop_size,
            file_path2,
            sheetname3,**kwargs

    ) -> None:
        super().__init__(
            n_var=time_window,
            n_obj=1,
            xl=[-1] * time_window,
            xu=[1] * time_window,
            vtype=float, **kwargs
        )

        self.data = Get_data.get_data(file_path2, sheetname3)
        self.PUN_timeseries = self.data.iloc[:time_window, 2].to_numpy()
        # self.c_d_timeseries = np.zeros((len(self.PUN_timeseries),self.pop_size))
        self.soc = np.zeros((len(self.PUN_timeseries)))
        self.charged_energy = np.zeros((len(self.PUN_timeseries)))
        self.discharged_energy = np.zeros((len(self.PUN_timeseries)))
        self.c_func, self.d_func = BESS.get_c_d_functions(load_curve)
        self.soc[0] = 0.2  # Inizializzazione della SOC

    def _evaluate(self, x, out, *args, **kwargs):
        self.c_d_timeseries = np.array(x).reshape(48)
        epsilon = 0  # Piccolo valore per confronto floating-point
        for index in range(len(self.PUN_timeseries) - 1):
            if self.c_d_timeseries[index] >= 0.0:
                self.c_d_timeseries[index] = min(self.c_d_timeseries[index],
                                                 min(self.c_func(self.soc[index]), 0.9 - self.soc[index]))
            else:
                self.c_d_timeseries[index] = max(self.c_d_timeseries[index],
                                                 max(-self.d_func(self.soc[index]), 0.1 - self.soc[index]))

            if self.c_d_timeseries[index] >= epsilon:
                self.charged_energy[index] = self.c_d_timeseries[index] * size
                self.discharged_energy[index] = 0.0
            elif self.c_d_timeseries[index] <= -epsilon:
                self.discharged_energy[index] = self.c_d_timeseries[index] * size
                self.charged_energy[index] = 0.0
            else:
                self.charged_energy[index] = 0.0
                self.discharged_energy[index] = 0.0

            # UPDATE SoC
            if self.c_d_timeseries[index] >= 0.0:
                self.soc[index + 1] = np.minimum(1, self.soc[index] + self.charged_energy[index] / size)
            else:
                self.soc[index + 1] = max(0.0, self.soc[index] + self.discharged_energy[index] / size)

        revenue_column= np.array(-(self.discharged_energy * self.PUN_timeseries / 1000)
        - (self.charged_energy * self.PUN_timeseries / 1000))

        total_revenue = sum(revenue_column)  # Somma lungo l'asse delle righe
        weighted_revenues = total_revenue  # Applica i pesi
        F_values = -total_revenue


        out["F"] = F_values
