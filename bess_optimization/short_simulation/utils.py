"""

BESS Optimization using NSGA-III Algorithm

    __author__ = "Lorenzo Giannuzzo"
    __maintainer__ = "Lorenzo Giannuzzo"
    __email__ = "lorenzo.giannuzzo@polito.it"
    __status__ = "in progress"
    __version__ = "v0.2.1"
    __license__ = "MIT"

Last Update of current code: 27/02/2025 """
import statistics

import ExcelOpener_s
import Interpolator_s
from pymoo.config import Config
Config.warnings['not_compiled'] = False

class Get_data:
    @staticmethod
    def get_data(file_path2, sheetname3):
        # Import data from an Excel file and a specific sheet using a method from the ExcelOpener module.
        data = ExcelOpener_s.import_file.load_excel(file_path2, sheetname3)
        # Return the imported data.
        return data

class BESS:

    @staticmethod
    def get_bess(technology, properties, se_sp, size):
        # Select specific columns from the properties DataFrame.
        BESS_Parameters = properties.iloc[:, [0, 1, 7, 9, 14, 15]]
        # Filter the DataFrame for the specified technology.
        BESS_Parameters = BESS_Parameters[BESS_Parameters['Technology'] == technology]
        # Further filter the DataFrame for the specified specific energy/specific power.
        BESS_Parameters = BESS_Parameters[BESS_Parameters['Specific Energy / Specific Power'] == se_sp]
        # Add a new column for the size in kWh.
        BESS_Parameters['Size [kWh]'] = size
        # Select specific columns to keep in the final DataFrame.
        BESS_Parameters = BESS_Parameters.iloc[:, [0, 1, 2, 3, 6]]

        # Return the filtered and modified DataFrame.
        return BESS_Parameters

    @staticmethod
    def get_c_d_functions(load_curve):

        # Select charge rate DataFrame from the first 356 rows of load_curve.
        charge_rate = load_curve.iloc[:356, [0, 3.5]]

        # Select discharge rate DataFrame from rows 357 onwards of load_curve.
        discharge_rate = load_curve.iloc[357:, [0, 4, 5]]

        # Interpolate data for the charge rate.
        charge_interpolator = Interpolator_s.DataInterpolator(charge_rate, 'SoC [%]', 'Charge Rate [kWh/(kWhp*h)]')
        charge_rate_interpolated_func = charge_interpolator.interpolate()

        # Interpolate data for the discharge rate.
        discharge_interpolator = Interpolator_s.DataInterpolator(discharge_rate, 'SoC [%]', 'Discharge Rate [kWh/(kWhp*h)]')
        discharge_rate_interpolated_func = discharge_interpolator.interpolate()

        # Return the interpolated functions for charge and discharge rates.
        return charge_rate_interpolated_func, discharge_rate_interpolated_func


def CustomSampling(energy_price, max_charge, max_discharge):

    import numpy as np

    energy_price = np.array(energy_price)
    mean_price = np.mean(energy_price)  # Use numpy's mean
    n = len(energy_price)

    # Initialize X with double length
    X = np.zeros(2 * n)

    # First half: Charge/discharge logic
    for i in range(n):
        if energy_price[i] > mean_price * 1.2:
            X[i] = max_discharge
        elif energy_price[i] < mean_price * 0.8:
            X[i] = max_charge
        else:
            X[i] = 0.0

    # Second half: All 1.0 values
    X[n:] = 1.0

    return X



