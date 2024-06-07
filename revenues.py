import numpy as np
import matplotlib
import Interpolator
import ExcelOpener
from pymoo.core.problem import Problem


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
se_sp = 10 # Specific Energy / Specific Power
size = 2500 #kWh

# LOADING FILES
Properties = Get_data.get_data(file_path, sheetname)
PUN = Get_data.get_data(file_path2,sheetname3)
load_curve = Get_data.get_data(file_path, sheetname2)

# Select charge_rate and discharge_rate DataFrames from load_curve
charge_rate = load_curve.iloc[:356, [0, 3,5]]
discharge_rate = load_curve.iloc[357:, [0, 4,5]]

# GET BESS PARAMETERS
BESS_Parameters = BESS.get_bess(technology,se_sp,size)

# GET C_D(SoC) FUNCTIONS
charge_rate_interpolated_func,discharge_rate_interpolated_func = BESS.get_c_d_functions(load_curve)

class Revenues(Problem):
    def __init__(
        self,
        size,
        pop_size,
        file_path2,
        sheetname3



    ) -> None:
        super().__init__(
            n_var=24,
            n_obj=1,
            xl=[-1] * 24,
            xu=[1] * 24,
            vtype=float
        )
        self.data = Get_data.get_data(file_path2, sheetname3)
        self.PUN_timeseries = self.data.iloc[:24, 2].to_numpy()
        self.c_d_timeseries = np.zeros(len(self.PUN_timeseries))
        self.soc = np.zeros(len(self.PUN_timeseries))
        self.charged_energy = np.zeros(len(self.PUN_timeseries))
        self.discharged_energy = np.zeros(len(self.PUN_timeseries))
        self.c_func, self.d_func = BESS.get_c_d_functions(load_curve)
        self.soc[0] = 0.2  # Inizializzazione della SOC

    def _evaluate(self, x, out, *args, **kwargs):
        self.c_d_timeseries = np.array(x).reshape(24, 1)


        for index in range(len(self.PUN_timeseries)-1):

            if self.c_d_timeseries[index] >= 0:
                self.c_d_timeseries[index] = min(self.c_d_timeseries[index], self.c_func(self.soc[index]))
            else:
                self.c_d_timeseries[index] = max(self.c_d_timeseries[index], -self.d_func(self.soc[index]))

            if self.c_d_timeseries[index] >= 0:
                self.charged_energy[index] = self.c_func(self.soc[index])*size
            else:
                self.discharged_energy[index] = -self.d_func(self.soc[index])*size

            # UPDATE SoC
            if self.c_d_timeseries[index] >= 0:
                self.soc[index + 1] = min(1, self.soc[index] + self.charged_energy[index]/size)
            else:
                self.soc[index + 1] = max(0, self.soc[index] + self.discharged_energy[index]/size)

        revenues = sum(-self.discharged_energy * self.PUN_timeseries) - sum(self.charged_energy * self.PUN_timeseries)
        out["F"] = np.array([-revenues.sum()])
        print(-revenues,self.charged_energy,self.discharged_energy)