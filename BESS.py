import matplotlib.pyplot as plt
import os
import numpy as np
from pymoo.algorithms.moo.nsga3 import NSGA3
from pymoo.optimize import minimize
from pymoo.util.ref_dirs import get_reference_directions
from revenues import Revenues
from pymoo.operators.sampling.rnd import FloatRandomSampling
from revenues import discharge_rate
from revenues import charge_rate
from revenues import PUN
from revenues import Get_data
from revenues import charge_rate_interpolated_func
from revenues import discharge_rate_interpolated_func
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM
from pymoo.operators.selection.tournament import TournamentSelection, compare
from pymoo.termination import get_termination



# FILE PATH DATA
file_path = "BESS Data.xlsx"
file_path2 = "PUN.xlsx"
sheetname = "BESS Properties"
sheetname2 = "Li-ion ChargeDischarge Curve 10"
sheetname3 = "PUN"

termination = get_termination("n_gen",10000)

def comp_by_cv_then_random(pop, P, **kwargs):
    S = np.full(P.shape[0], np.nan)

    for i in range(P.shape[0]):
        a, b = P[i, 0], P[i, 1]

        # if at least one solution is infeasible
        if pop[a].CV > 0.0 or pop[b].CV > 0.0:
            S[i] = compare(a, pop[a].CV, b, pop[b].CV, method='smaller_is_better', return_random_if_equal=True)

        # both solutions are feasible just set random
        else:
            S[i] = np.random.choice([a, b])

    return S[:, None].astype(int)


class Optimizer:

    def __init__(self, objective_function: Revenues, pop_size: int):
        self._objective_function = objective_function
        self.pop_size = pop_size


    def maximize_revenues(self):
        ref_dirs = get_reference_directions("das-dennis", 1,n_partitions=20) # n_partitions=48

        # Creare l'algoritmo con gli operatori definiti
        algorithm = NSGA3(
            pop_size=self.pop_size,
            ref_dirs=ref_dirs,
            sampling=FloatRandomSampling(),
            selection=TournamentSelection(func_comp=comp_by_cv_then_random),
            eliminate_duplicates=True,
            crossover=SBX(eta=30, prob=1.0),
            mutation=PM(eta=20),

        )

        res = minimize(
            self._objective_function,
            algorithm,
            termination,
            seed=42,
            verbose=True,
        )

        return res

# [OPTIMIZATION]
pop_size = 1
size = 2500
objective_function = Revenues(size=size, pop_size=pop_size, file_path2=file_path2, sheetname3=sheetname3)

# Create an instance of Optimizer with the objective_function and pop_size
optimizer = Optimizer(objective_function=objective_function, pop_size=pop_size)

# Maximize revenues
solution = optimizer.maximize_revenues()

soc = [0.0] * 48
charged_energy = [0.0]*48
discharged_energy= [0.0]*48
c_d_timeseries = solution.X
soc[0] = 0.20

c_func = charge_rate_interpolated_func
d_func = discharge_rate_interpolated_func

for index in range(48 - 1):
    if c_d_timeseries[index] >= 0:
        c_d_timeseries[index] = min(c_d_timeseries[index], min(c_func(soc[index]), 0.9-soc[index]))
    else:
        c_d_timeseries[index] = max(c_d_timeseries[index], max(-d_func(soc[index]), -soc[index]+0.1))

    if c_d_timeseries[index] >= 0:
        charged_energy[index] = c_d_timeseries[index] * size
    else:
        discharged_energy[index] = c_d_timeseries[index] * size

        # UPDATE SoC
    if c_d_timeseries[index] >= 0:
        soc[index + 1] = min(0.9, soc[index] + charged_energy[index]/ size)
    else:
        soc[index + 1] = max(0.1, soc[index] + discharged_energy[index]/ size)

data = Get_data.get_data(file_path2, sheetname3)
PUN_timeseries = data.iloc[:48,2].to_numpy()

rev = - (discharged_energy* PUN_timeseries/1000) - (charged_energy*PUN_timeseries/1000)
print(rev.sum())

# Creazione degli istogrammi
time_steps = np.arange(48)

# Istogramma di SoC
plt.figure(figsize=(12, 8))
plt.bar(time_steps, soc, color='lightblue')
plt.title('State of Charge (SoC) [%]')
plt.xlabel('Time step')
plt.ylabel('SoC')
plt.savefig(os.path.join("Plots", "SoC_hist.png"))
plt.close()


# Istogramma di charged energy
plt.figure(figsize=(12, 8))
plt.bar(time_steps, charged_energy, color='g')
plt.title('Charged Energy')
plt.xlabel('Time step')
plt.ylabel('Charged Energy')
plt.savefig(os.path.join("Plots", "Charged_Energy_hist.png"))
plt.close()

# Istogramma di discharged energy
plt.figure(figsize=(12, 8))
plt.bar(time_steps, discharged_energy, color='r')
plt.title('Discharged Energy')
plt.xlabel('Time step')
plt.ylabel('Discharged Energy')
plt.savefig(os.path.join("Plots", "Discharged_Energy_hist.png"))
plt.close()


# Line plot dei valori PUN (terza colonna del DataFrame)
pun_values = PUN.iloc[:48, 2]  # Estrazione della terza colonna (indice 2)
plt.figure(figsize=(12, 8))
plt.plot(pun_values, marker='o', color='b')
plt.title('PUN Values')
plt.xlabel('Time step')
plt.ylabel('PUN Value')
plt.savefig(os.path.join("Plots", "PUN_values_plot.png"))
plt.close()

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

num_values = 48
time_steps_24 = time_steps[:num_values]
charged_energy_24 = charged_energy[:num_values]
discharged_energy_24 = discharged_energy[:num_values]
pun_values_24 = PUN.iloc[:num_values, 2]

# Istogramma combinato di charged energy e discharged energy
fig, ax1 = plt.subplots(figsize=(12, 8))
width = 0.4
ax1.bar(time_steps_24 - width / 2, charged_energy_24, width=width, color='g', label='Charged Energy [kWh]')
ax1.bar(time_steps_24 + width / 2, discharged_energy_24, width=width, color='r', label='Discharged Energy [kWh]')
ax1.set_xlabel('Time step')
ax1.set_ylabel('Energy [kWh]')
ax1.set_title('Charged and Discharged Energy with PUN')
ax1.legend(loc='upper left')

# Aggiungi un secondo asse y per i valori PUN
ax2 = ax1.twinx()
ax2.plot(time_steps_24, pun_values_24, marker='o', color='navy', label='PUN [Euro/MWh]')
ax2.set_ylabel('PUN Value')
ax2.legend(loc='upper right')

fig.tight_layout()
plt.savefig(os.path.join("Plots", "Charged_and_Discharged_Energy_with_PUN_24.png"))
plt.close()


