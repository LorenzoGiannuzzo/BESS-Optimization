import configuration
from pymoo.optimize import minimize
from pymoo.core.problem import StarmapParallelization
from objective_function import Revenues
import multiprocessing
from multiprocessing import cpu_count


class Optimizer:
    def __init__(self, objective_function: Revenues, pop_size: int):
        self._objective_function = objective_function
        self.pop_size = pop_size

    def maximize_revenues(self):
        n_processes = 8  # Set the number of processes
        pool = multiprocessing.Pool(processes=n_processes)
        runner = StarmapParallelization(pool.starmap)

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
            elementwise_evaluator=runner
        )
        print('Execution Time:', res.exec_time)

        pool.close()
        pool.join()

        return res






