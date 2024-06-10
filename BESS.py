import matplotlib.pyplot as plt
from pymoo.algorithms.moo.nsga3 import NSGA3
from pymoo.optimize import minimize
from pymoo.util.ref_dirs import get_reference_directions
from revenues import Revenues

# FILE PATH DATA
file_path = "BESS Data.xlsx"
file_path2 = "PUN.xlsx"
sheetname = "BESS Properties"
sheetname2 = "Li-ion ChargeDischarge Curve 10"
sheetname3 = "PUN"

class Optimizer:

    def __init__(self, objective_function: Revenues, pop_size: int) -> None:
        self._objective_function = objective_function
        self.pop_size = pop_size
    def maximize_revenues(self):
        ref_dirs = get_reference_directions("das-dennis", 1, n_partitions=12)
        algorithm = NSGA3(pop_size=self.pop_size, ref_dirs=ref_dirs)

        res = minimize(
            self._objective_function,
            algorithm,
            ('n_gen', 600),
            seed=42,
            verbose=True,
        )
        return res

# [OPTIMIZATION]
pop_size = 1
size= 2500
objective_function = Revenues(size=size, pop_size= pop_size,file_path2="PUN.xlsx", sheetname3="PUN")

# Create an instance of Optimizer with the objective_function and pop_size
optimizer = Optimizer(objective_function=objective_function, pop_size=pop_size)

# Maximize revenues
solution = optimizer.maximize_revenues()




#
#
# # Plotting
# plt.figure(figsize=(10, 6))
#
# # Plot for charge_rate
# plt.plot(charge_rate['SoC [%]'], charge_rate['Charge Rate [kWh/(kWhp*h)]'], 'o', label='Charge Rate')
# plt.plot(charge_rate['SoC [%]'], charge_rate_interpolated_func(charge_rate['SoC [%]']), '-',
#          label='Interpolated Charge Rate')
#
# # Plot for discharge_rate
# plt.plot(discharge_rate['SoC [%]'], discharge_rate['Discharge Rate [kWh/(kWhp*h)]'], 'o', color='red',
#          label='Discharge Rate')
# plt.plot(discharge_rate['SoC [%]'], discharge_rate_interpolated_func(discharge_rate['SoC [%]']), '-',
#          color='green', label='Interpolated Discharge rate')
#
# plt.xlabel('SoC [%]')
# plt.ylabel('Rate [kWh/(kWhp*h)]')
# plt.title('Interpolated Functions')
# plt.legend()
# plt.grid(True)
#
# # Save the plot as a PNG file
# if not os.path.exists("Plots"):
#     os.makedirs("Plots")
# plt.savefig("Plots/interpolated_functions.png")
#
# # Close the figure to release memory
# plt.close()
#
# # Plotting for charge_rate
# plt.figure(figsize=(10, 6))
# plt.plot(charge_rate['SoC [%]'], charge_rate['Charge Rate [kWh/(kWhp*h)]'], label='Charge Rate')
# plt.xlabel('SoC [%]')
# plt.ylabel('Charge Rate [kWh/(kWhp*h)]')
# plt.title('Charge Rate vs SoC')
# plt.legend()
# plt.grid(True)
#
# # Save the plot as a PNG file
# plt.savefig("Plots/charge_rate_plot.png")
# plt.close()
#
# # Plotting for discharge_rate
# plt.figure(figsize=(10, 6))
# plt.plot(discharge_rate['SoC [%]'], discharge_rate['Discharge Rate [kWh/(kWhp*h)]'], color='red',
#          label='Discharge Rate')
# plt.xlabel('SoC [%]')
# plt.ylabel('Discharge Rate [kWh/(kWhp*h)]')
# plt.title('Discharge Rate vs SoC')
# plt.legend()
# plt.grid(True)
#
# # Save the plot as a PNG file
# plt.savefig("Plots/discharge_rate_plot.png")
# plt.close()




