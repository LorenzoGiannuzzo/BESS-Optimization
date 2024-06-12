import matplotlib.pyplot as plt
import os
import numpy as np
from pymoo.algorithms.moo.nsga3 import NSGA3
from pymoo.optimize import minimize
from pymoo.util.ref_dirs import get_reference_directions
from pymoo.operators.sampling.rnd import FloatRandomSampling
from objective_function import Revenues
from objective_function import pop_size
from objective_function import termination
from Economic_parameters import PUN_timeseries
from BESS_parameters import charge_rate_interpolated_func, discharge_rate_interpolated_func
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM
from pymoo.operators.selection.tournament import TournamentSelection, compare
from BESS_parameters import size, file_path, file_path2, sheetname, sheetname2, sheetname3
from objective_function import time_window






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

    def __init__(self, objective_function: Revenues, pop_size: pop_size):
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
            termination=termination,
            seed=42,
            verbose=True,
        )

        return res

# [OPTIMIZATION]
# import multiprocessing
# from pymoo.algorithms.soo.nonconvex.ga import GA
# from pymoo.core.problem import StarmapParallelization
#
# # initialize the thread pool and create the runner
# n_proccess = 2
# pool = multiprocessing.Pool(n_proccess)
# runner = StarmapParallelization(pool.starmap)


objective_function = Revenues(size=size, pop_size=pop_size, file_path2=file_path2, sheetname3=sheetname3) #elementwise_runner=runner)

# Create an instance of Optimizer with the objective_function and pop_size
optimizer = Optimizer(objective_function=objective_function, pop_size=pop_size)

# Maximize revenues
solution = optimizer.maximize_revenues()
#pool.close()

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

PUN_timeseries = PUN_timeseries

rev = - (discharged_energy * PUN_timeseries/1000) - (charged_energy*PUN_timeseries/1000)
print("Revenus for optimized time window:\n", rev.sum())

import matplotlib
matplotlib.use('Agg')

# Creazione degli istogrammi
time_steps = np.arange(time_window)

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



num_values = time_window
time_steps_24 = time_steps[:num_values]
charged_energy_24 = charged_energy[:num_values]
discharged_energy_24 = discharged_energy[:num_values]
pun_values_24 = PUN_timeseries[:num_values]

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


