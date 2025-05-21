"""

BESS Optimization using NSGA-III Algorithm

    __author__ = "Lorenzo Giannuzzo"
    __maintainer__ = "Lorenzo Giannuzzo"
    __email__ = "lorenzo.giannuzzo@polito.it"
    __status__ = "in progress"
    __version__ = "v0.2.1"
    __license__ = "MIT"

Last Update of current code: 09/01/2025 - 17:49

"""

import ExcelOpener_s
import Interpolator_s

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


import pandas as pd

# Define the path to your CSV file
file_path = r'C:\Users\lorenzo.giannuzzo\PycharmProjects\BESS-Optimization\data\Input\pv\year_PV.csv'

# Load the CSV file into a DataFrame
df = pd.read_csv(file_path, sep=";")

df['P'] = df['P'] * 10

# Save the updated DataFrame back to the CSV file (overwrite)
df.to_csv(r'C:\Users\lorenzo.giannuzzo\PycharmProjects\BESS-Optimization\data\Input\pv\REC_PV.csv', index=False, sep=";")