import configuration

from multiprocessing.pool import ThreadPool
from pymoo.optimize import minimize
from pymoo.core.problem import StarmapParallelization
from objective_function import Revenues
import multiprocessing
from multiprocessing import cpu_count


class Optimizer:
    def __init__(self, objective_function: Revenues, pop_size: int, multiprocessing = True):
        self._objective_function = objective_function
        self.pop_size = pop_size
        self.multiprocessing = multiprocessing

    def maximize_revenues(self):
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
                save_history=True,

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






