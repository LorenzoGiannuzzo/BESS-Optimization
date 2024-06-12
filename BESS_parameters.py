from utils import Get_data
from utils import BESS
import matplotlib.pyplot as plt
import os


# FILE PATH DATA
file_path = "BESS Data.xlsx"
file_path2 = "PUN.xlsx"
sheetname = "BESS Properties"
sheetname2 = "Li-ion ChargeDischarge Curve 10"
sheetname3 = "PUN"

'''
DEFINE BESS PARAMETERS:
   
   1) Technology
   2) Nominal Capacity / Nominal Power (Specific Energy / Specific Power)
   3) Size

'''

# BESS Parameters

technology = "Li-ion"     #Define BESS Technology
se_sp = 10   #Define Nominal Capacity / Nominal Power - Parameter
size = 2500  #Define BESS size (Nominal Capacity) in kWh

# LOADING FILES

properties = Get_data.get_data(file_path, sheetname)
PUN = Get_data.get_data(file_path2, sheetname3)
load_curve = Get_data.get_data(file_path, sheetname2)


# TAKE CHARGE AND DISCHARGE BATTERY LOAD CURVES (to do: still to be automatized)

charge_rate = load_curve.iloc[:356, [0, 3, 5]]
discharge_rate = load_curve.iloc[357:, [0, 4, 5]]


# GET CHOSEN BESS PARAMETERS

BESS_Parameters = BESS.get_bess(technology, properties, se_sp, size)


# INTERPOLATE CHARGE AND DISCHARGE FUNCTIONS FROM CHARGE AND DISCHARGE RATE

charge_rate_interpolated_func, discharge_rate_interpolated_func = BESS.get_c_d_functions(load_curve)

import matplotlib
matplotlib.use('Agg')

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