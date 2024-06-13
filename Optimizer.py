import numpy as np

from pymoo.algorithms.moo.nsga3 import NSGA3
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM
from pymoo.operators.sampling.rnd import FloatRandomSampling
from pymoo.operators.selection.tournament import TournamentSelection, compare
from pymoo.optimize import minimize
from pymoo.util.ref_dirs import get_reference_directions

from BESS_parameters import charge_rate_interpolated_func, discharge_rate_interpolated_func
from BESS_parameters import size

from Economic_parameters import PUN_timeseries

from objective_function import Revenues
from objective_function import pop_size
from objective_function import termination
from objective_function import time_window
from objective_function import soc_0

from Plots import EnergyPlots

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
            save_history = True
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

