import configuration
import multiprocessing

from pymoo.optimize import minimize
from pymoo.core.problem import StarmapParallelization
from objective_function import Revenues
from multiprocessing import cpu_count
from configuration import plot


class Optimizer:
    def __init__(self, objective_function: Revenues, pop_size: int, multiprocessing = True):
        self._objective_function = objective_function
        self.pop_size = pop_size
        self.multiprocessing = multiprocessing

    def maximize_revenues(self):
        if plot:
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
            print('Execution Time:', res.exec_time)

        return res






