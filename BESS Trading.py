'''

STANDARD LIBRARIES

'''

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import os
import ExcelOpener
import Interpolator
import math

'''

OPTIMIZATION LIBRARIES

'''

from pymoo.algorithms.moo.nsga3 import NSGA3
from pymoo.optimize import minimize
from pymoo.problems import get_problem
from pymoo.util.ref_dirs import get_reference_directions
from pymoo.visualization.scatter import Scatter

# Set matplotlib backend to 'Agg' for non-interactive plotting
matplotlib.use('Agg')

# Load BESS Data from Excel Dataframe
file_path = "BESS Data.xlsx"
file_path2 = "PUN.xlsx"
sheetname = "BESS Properties"
sheetname2 = "Li-ion ChargeDischarge Curve 10"
sheetname3 = "PUN"
Properties = ExcelOpener.import_file.load_excel(file_path, sheetname)
load_curve = ExcelOpener.import_file.load_excel(file_path, sheetname2)
PUN = ExcelOpener.import_file.load_excel(file_path2, sheetname3)

# Select charge_rate and discharge_rate DataFrames from load_curve
charge_rate = load_curve.iloc[:356, [0, 3.5]]
discharge_rate = load_curve.iloc[357:, [0, 4,5]]

''' BESS SETUP

    - Technology
    - Size [kWh]
    - Specific Energy / Specific Power
    - Initial SoC

'''

technology = "Li-ion"
se_sp = 10 # Specific Energy / Specific Power
weight = 100 #[kg]
SoC_0 = 0.20 #[%]

''' BESS Parameter Evaluation

    - Self Dischage [%/a]
    - Efficiency
    - Specific Energy / Specific Power

'''

# Extract relevant columns from Properties DataFrame and filter based on technology and specific energy/specific power
BESS_Parameters = Properties.iloc[:,[0,1,7,9,14,15]]
BESS_Parameters = BESS_Parameters[BESS_Parameters['Technology'] == technology]
BESS_Parameters = BESS_Parameters[BESS_Parameters['Specific Energy / Specific Power'] == se_sp]

# Calculate Size [kWh] based on Upper Bound Specific Energy [Wh/kg] and weight, then select relevant columns
BESS_Parameters['Size [kWh]'] = BESS_Parameters['Upper Bound Specific Energy [Wh/kg]']*weight/1000
BESS_Parameters = BESS_Parameters.iloc[:,[0,1,2,3,6]]

''' OPTIMIZATION PROBLEM STATEMENT

    Optimizaiton Problem Definitizion:
     
     - Optimization Algorithm (NSGA-III)
     - Dynamic Parameter Evaluation

'''
# Interpolate data for charge_rate and discharge_rate
charge_interpolator = Interpolator.DataInterpolator(charge_rate, 'SoC [%]', 'Charge Rate [kWh/(kWhp*h)]')
charge_rate_interpolated_func = charge_interpolator.interpolate()

discharge_interpolator = Interpolator.DataInterpolator(discharge_rate, 'SoC [%]', 'Discharge Rate [kWh/(kWhp*h)]')
discharge_rate_interpolated_func = discharge_interpolator.interpolate()






# Plotting
plt.figure(figsize=(10, 6))

# Plot for charge_rate
plt.plot(charge_rate['SoC [%]'], charge_rate['Charge Rate [kWh/(kWhp*h)]'], 'o', label='Charge Rate')
plt.plot(charge_rate['SoC [%]'], charge_rate_interpolated_func(charge_rate['SoC [%]']), '-',
         label='Interpolated Charge Rate')

# Plot for discharge_rate
plt.plot(discharge_rate['SoC [%]'], discharge_rate['Discharge Rate [kWh/(kWhp*h)]'], 'o', color='red',
         label='Discharge Rate')
plt.plot(discharge_rate['SoC [%]'], discharge_rate_interpolated_func(discharge_rate['SoC [%]']), '-',
         color='green', label='Interpolated Discharge rate')

plt.xlabel('SoC [%]')
plt.ylabel('Rate [kWh/(kWhp*h)]')
plt.title('Interpolated Functions')
plt.legend()
plt.grid(True)

# Save the plot as a PNG file
if not os.path.exists("Plots"):
    os.makedirs("Plots")
plt.savefig("Plots/interpolated_functions.png")

# Close the figure to release memory
plt.close()

# Plotting for charge_rate
plt.figure(figsize=(10, 6))
plt.plot(charge_rate['SoC [%]'], charge_rate['Charge Rate [kWh/(kWhp*h)]'], label='Charge Rate')
plt.xlabel('SoC [%]')
plt.ylabel('Charge Rate [kWh/(kWhp*h)]')
plt.title('Charge Rate vs SoC')
plt.legend()
plt.grid(True)

# Save the plot as a PNG file
plt.savefig("Plots/charge_rate_plot.png")
plt.close()

# Plotting for discharge_rate
plt.figure(figsize=(10, 6))
plt.plot(discharge_rate['SoC [%]'], discharge_rate['Discharge Rate [kWh/(kWhp*h)]'], color='red',
         label='Discharge Rate')
plt.xlabel('SoC [%]')
plt.ylabel('Discharge Rate [kWh/(kWhp*h)]')
plt.title('Discharge Rate vs SoC')
plt.legend()
plt.grid(True)

# Save the plot as a PNG file
plt.savefig("Plots/discharge_rate_plot.png")
plt.close()




