"""

BESS Optimization using NSGA-III Algorithm

    __author__ = "Lorenzo Giannuzzo"
    __maintainer__ = "Lorenzo Giannuzzo"
    __email__ = "lorenzo.giannuzzo@polito.it"
    __status__ = "in progress"
    __version__ = "v0.2.1"
    __license__ = "MIT"

Last Update of current code: 09/01/2025 - 17:38

"""

# IMPORT LIBRARIES
import configuration
from pymoo.optimize import minimize
from objective_function import Revenues
from configuration import plot


# DEFINE OPTIMIZER CLASS
class Optimizer:

    def __init__(self, objective_function: Revenues, pop_size: int, multiprocessing=True):

        # MULTIPROCESSING CAN BE DISABLED TO COMPARE ALGORITHMS EXECUTION TIMES
        self._objective_function = objective_function
        self.pop_size = pop_size
        self.multiprocessing = multiprocessing

    # DEFINE THE OPTIMIZATION TASK: MAXIMIZATION OF REVENUES
    def maximize_revenues(self):

        if plot:
            # SAVE OPTIMIZATION HISTORY IF PLOTS ARE REQUIRED
            history = True
        else:
            history = False

        if self.multiprocessing:

            problem = self._objective_function
            algorithm = configuration.algorithm
            termination = configuration.termination

            res = minimize(

                problem,
                algorithm,
                termination,
                seed=42,
                verbose=True,
                save_history=history,

            )

            # VISUALIZE EXECUTION TIME
            print('Execution Time:', res.exec_time)

        else:

            problem = self._objective_function
            algorithm = configuration.algorithm
            termination = configuration.termination

            res = minimize(
                problem,
                algorithm,
                termination,
                seed=42,
                verbose=True,
                save_history=True,
            )

            # VISUALIZE EXECUTION TIME
            print('Execution Time:', res.exec_time)

        return res






