import configuration

from pymoo.optimize import minimize
from objective_function import Revenues

# OPTIMIZER CREATION

class Optimizer:

    def __init__(self, objective_function: Revenues, pop_size: configuration.pop_size):
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

        # ALGORITHM AND HYPERPARAMETERS DEFINITION

        algorithm = configuration.algorithm

        # INITIALIZE THE RESOLUTION OF THE OPTIMIZATION PROBLEM

        res = minimize(
            self._objective_function,
            algorithm,
            termination=configuration.termination,
            seed=42,
            verbose=True,
            save_history=True
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

