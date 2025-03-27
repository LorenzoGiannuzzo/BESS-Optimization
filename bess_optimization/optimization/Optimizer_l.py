""" BESS Optimization using NSGA-III Algorithm

    __author__ = "Lorenzo Giannuzzo"
    __maintainer__ = "Lorenzo Giannuzzo"
    __email__ = "lorenzo.giannuzzo@polito.it"
    __status__ = "in progress"
    __version__ = "v0.2.1"
    __license__ = "MIT"

Last Update of current code: 24/03/2025 """

# IMPORT LIBRARIES AND MODULES
import configuration_l
import numpy as np
from pymoo.optimize import minimize
from objective_function_l import Revenues
from configuration_l import plot
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM
from pymoo.optimize import minimize

from pymoo.termination import get_termination
from pymoo.problems import get_problem


class CustomCallback:
    def __init__(self, update_interval):
        self.update_interval = update_interval
        self.new_eta_crossover = configuration_l.eta_crossover
        self.new_prob_crossover = configuration_l.prob_crossover
        self.new_eta_mutation = configuration_l.eta_mutation
        self.new_prob_mutation = configuration_l.prob_mutation

    def calculate_diversity(self, population):
        # Calculate the pairwise Euclidean distance between solution vectors
        N = population.shape[0]
        diversity_sum = 0.0

        for i in range(N):
            for j in range(i + 1, N):  # Only compute for unique pairs
                distance = np.linalg.norm(population[i] - population[j])
                diversity_sum += distance

        # Average distance across all unique pairs
        diversity = (2 * diversity_sum) / (N * (N - 1))
        return diversity

    def __call__(self, algorithm):
        generation = algorithm.n_gen
        if generation % self.update_interval == 0:
            # Get the population
            population = algorithm.pop.get("X")

            # Calculate the diversity using the entire solution vectors
            diversity = self.calculate_diversity(population)

            # Update parameters based on diversity
            if diversity < 0.1:  # If diversity is low, increase eta
                self.new_eta_crossover = np.minimum(1.5 * self.new_eta_crossover, 20.0)
                self.new_eta_mutation = np.minimum(1.5 * self.new_eta_mutation, 30.0)
            else:  # If diversity is high, decrease eta to encourage convergence
                self.new_eta_crossover = np.maximum(0.5 * self.new_eta_crossover, 1.0)
                self.new_eta_mutation = np.maximum(0.5 * self.new_eta_mutation, 3.0)


            # Create new instances of crossover and mutation with updated parameters
            algorithm.mating.crossover = SBX(eta=self.new_eta_crossover, prob=self.new_prob_crossover)
            algorithm.mating.mutation = PM(eta=self.new_eta_mutation, prob=self.new_prob_mutation)

            print(
                f"Generation: {generation}, New eta Crossover: {self.new_eta_crossover}, New eta Mutation: {self.new_eta_mutation}, Diversity: {diversity}")
    def calculate_diversity(self, population):
        # Calculate diversity as the average distance between individuals
        distances = np.linalg.norm(population[:, np.newaxis] - population, axis=2)
        diversity = np.mean(distances)
        return diversity

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

        callback = CustomCallback(20)

        if self.multiprocessing:

            problem = self._objective_function
            algorithm = configuration_l.algorithm
            termination = configuration_l.termination

            res = minimize(
                problem,
                algorithm,
                termination,
                seed=42,
                verbose=True,
                save_history=history,
                callback=callback
            )

            # VISUALIZE EXECUTION TIME
            print('Execution Time:', res.exec_time)

        else:
            problem = self._objective_function
            algorithm = configuration_l.algorithm
            termination = configuration_l.termination

            res = minimize(
                problem,
                algorithm,
                termination,
                seed=42,
                verbose=True,
                save_history=True,
                callback = callback
            )

            # VISUALIZE EXECUTION TIME
            print('Execution Time:', res.exec_time)

        return res

