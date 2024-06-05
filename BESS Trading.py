import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import  matplotlib
import os
import ExcelOpener
import Interpolator

matplotlib.use('Agg')
# Get BESS Data from Excel Dataframe

file_path = "BESS Data.xlsx"
file_path2 = "PUN.xlsx"
sheetname = "BESS Properties"
sheetname2 = "Li-ion ChargeDischarge Curve"
Properties = ExcelOpener.import_file.load_excel(file_path, sheet_name=sheetname)
load_curve = ExcelOpener.import_file.load_excel((file_path, sheet_name=sheetname2)
PUN = ExcelOpener.import_file.load_excel(file_path2,

# Select charge_rate and discharge_rate DataFrames from load_curve

charge_rate = load_curve.iloc[:356, [0, 3]]
discharge_rate = load_curve.iloc[357:, [0, 4]]

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
plt.plot(discharge_rate['SoC [%]'], discharge_rate_interpolated_func(discharge_rate['SoC [%]']), '-', color='green',
             label='Interpolated Discharge rate')

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

# Plotting charge_rate

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

# Plotting discharge_rate

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







