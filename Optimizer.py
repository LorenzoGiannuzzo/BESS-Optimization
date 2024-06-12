import os
import matplotlib.pyplot as plt
import numpy as np

from pymoo.algorithms.moo.nsga3 import NSGA3
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM
from pymoo.operators.sampling.rnd import FloatRandomSampling
from pymoo.operators.selection.tournament import TournamentSelection, compare
from pymoo.optimize import minimize
from pymoo.util.ref_dirs import get_reference_directions

from BESS_parameters import charge_rate_interpolated_func, discharge_rate_interpolated_func
from BESS_parameters import size, file_path2, sheetname3

from Economic_parameters import PUN_timeseries

from objective_function import Revenues
from objective_function import pop_size
from objective_function import termination
from objective_function import time_window
from objective_function import soc_0


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


# OPTIMIZER DEFINITION

class Optimizer:

    def __init__(self, objective_function: Revenues, pop_size: pop_size):
        self._objective_function = objective_function
        self.pop_size = pop_size

    # ALGORITHM DEFINITION

    def maximize_revenues(self):

        # DEFINE REFERENCE DIRECTION

        '''

        Most studies have used the Das and Dennis’s structured approach for generating well-spaced reference points.
        A reference direction is constructed by a vector originating from the origin and connected to each of them. The number
        of points on the unit simplex is determined by a parameter p (we call it n_partitions in our implementation), which
        indicates the number of gaps between two consecutive points along an objective axis.

        '''

        ref_dirs = get_reference_directions("das-dennis", 1, n_partitions=20)

        # ALGORITHM AND HYPERPARAMETERS DEFINITION

        algorithm = NSGA3(

            pop_size=self.pop_size,
            ref_dirs=ref_dirs,

            # sampling: This parameter specifies the method used to initialize the population. FloatRandomSampling
            # generates random floating-point values for the initial solutions, providing a diverse starting point.

            sampling=FloatRandomSampling(),

            # selection: This defines the selection mechanism used to choose parents for reproduction.
            # TournamentSelection selects individuals based on a comparison function.
            # func_comp=comp_by_cv_then_random: This comparison function first considers constraint violations (cv) and
            # then applies a random selection if necessary. This helps prioritize feasible solutions.

            selection=TournamentSelection(func_comp=comp_by_cv_then_random),

            # crossover: This parameter specifies the crossover operator used for generating offspring.
            # SBX: Simulated Binary Crossover (SBX) is a common crossover method in genetic algorithms, particularly
            # suited for real-valued variables.
            # eta=30: The distribution index for SBX. A higher value of eta results in offspring closer to their
            # parents, while a lower value results in more variation.
            # prob=1.0: The probability of crossover being applied. A probability of 1.0 means crossover is always
            # applied.

            crossover=SBX(eta=30, prob=1.0),

            # mutation: This parameter specifies the mutation operator used for generating variation in offspring.
            # PM: Polynomial Mutation (PM) is a common mutation method for real-valued variables.
            # eta=20: The distribution index for PM. Similar to SBX, a higher value of eta results in smaller mutations,
            # while a lower value results in larger mutations.

            mutation=PM(eta=20),

            eliminate_duplicates=True
        )

        # INITIALIZE THE RESOLUTION OF THE OPTIMIZATION PROBLEM

        res = minimize(
            self._objective_function,
            algorithm,
            termination=termination,
            seed=42,
            verbose=True,
        )

        return res

# to do: parallelize the optimization problem resolution

# import multiprocessing
# from pymoo.algorithms.soo.nonconvex.ga import GA
# from pymoo.core.problem import StarmapParallelization
#
# # initialize the thread pool and create the runner
# n_proccess = 2
# pool = multiprocessing.Pool(n_proccess)
# runner = StarmapParallelization(pool.starmap)
# ...
# ...
#  pool.close()


# STATEMENT OF THE OBJECTIVE FUNCTION

objective_function = Revenues(size=size, pop_size=pop_size, file_path2=file_path2,
                              sheetname3=sheetname3)  # elementwise_runner=runner)


# LAUNCH THE OPTIMIZER

optimizer = Optimizer(objective_function=objective_function, pop_size=pop_size)

# GET SOLUTION OF THE OPTIMIZATION PROBLEM

solution = optimizer.maximize_revenues()

# GET THE BEST X-ARRAY FROM SOLUTIONS AND RE-APPLY THE PHYSICAL CONSTRAINTS OF THE BESS
c_d_timeseries = solution.X

soc = [0.0] * time_window
charged_energy = [0.0] * time_window
discharged_energy = [0.0] * time_window
soc[0] = soc_0

c_func = charge_rate_interpolated_func
d_func = discharge_rate_interpolated_func

for index in range(48 - 1):
    if c_d_timeseries[index] >= 0:
        c_d_timeseries[index] = min(c_d_timeseries[index], min(c_func(soc[index]), 0.9 - soc[index]))
    else:
        c_d_timeseries[index] = max(c_d_timeseries[index], max(-d_func(soc[index]), -soc[index] + 0.1))

    if c_d_timeseries[index] >= 0:
        charged_energy[index] = c_d_timeseries[index] * size
    else:
        discharged_energy[index] = c_d_timeseries[index] * size

        # UPDATE SoC
    if c_d_timeseries[index] >= 0:
        soc[index + 1] = min(0.9, soc[index] + charged_energy[index] / size)
    else:
        soc[index + 1] = max(0.1, soc[index] + discharged_energy[index] / size)

PUN_timeseries = PUN_timeseries

rev = - (discharged_energy * PUN_timeseries / 1000) - (charged_energy * PUN_timeseries / 1000)
print("Revenus for optimized time window [Euros]:\n", rev.sum())





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
