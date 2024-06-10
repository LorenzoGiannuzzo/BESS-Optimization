import matplotlib.pyplot as plt
import pymoo
from pymoo.algorithms.moo.nsga3 import NSGA3
from pymoo.optimize import minimize
from pymoo.util.ref_dirs import get_reference_directions
from revenues import Revenues
from pymoo.operators.mutation.pm import PolynomialMutation
from pymoo.operators.crossover.sbx import SimulatedBinaryCrossover
from pymoo.operators.selection.tournament import TournamentSelection

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
        ref_dirs = get_reference_directions("das-dennis", 1, n_partitions=24)
        # Definire gli operatori di mutazione, crossover e selezione

        # Creare l'algoritmo con gli operatori definiti
        algorithm = NSGA3(pop_size=self.pop_size, ref_dirs=ref_dirs,
            )

        res = minimize(
            self._objective_function,
            algorithm,
            ('n_gen', 500),
            seed=42,
            verbose=True,

        )
        return res

# [OPTIMIZATION]
pop_size = 10
size= 2500
objective_function = Revenues(size=size, pop_size= pop_size,file_path2="PUN.xlsx", sheetname3="PUN")

# Create an instance of Optimizer with the objective_function and pop_size
optimizer = Optimizer(objective_function=objective_function, pop_size=pop_size)

# Maximize revenues
solution = optimizer.maximize_revenues()

soc = [0.0] * 48
charged_energy = [0.0]*48
discharged_energy= [0.0]*48
c_d_timeseries = solution.X
soc[0] = 0.2

from revenues import charge_rate_interpolated_func
from revenues import discharge_rate_interpolated_func
c_func = charge_rate_interpolated_func
d_func = discharge_rate_interpolated_func

for index in range(48 - 1):
    if c_d_timeseries[index] >= 0:
        c_d_timeseries[index] = min(c_d_timeseries[index], c_func(soc[index]), 0.9-soc[index])
    else:
        c_d_timeseries[index] = max(c_d_timeseries[index], -d_func(soc[index]), -soc[index]+0.1)

    if c_d_timeseries[index] >= 0:
        charged_energy[index] = c_d_timeseries[index] * size
    else:
        discharged_energy[index] = c_d_timeseries[index] * size

        # UPDATE SoC
    if c_d_timeseries[index] >= 0:
        soc[index + 1] = min(1, soc[index] + charged_energy[index] / size)
    else:
        soc[index + 1] = max(0, soc[index] + discharged_energy[index] / size)


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




