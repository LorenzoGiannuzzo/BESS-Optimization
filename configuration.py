import numpy as np

from argparser import soc
from Economic_parameters import time_window
from pymoo.termination.robust import RobustTermination
from pymoo.termination.xtol import DesignSpaceTermination
from pymoo.algorithms.moo.nsga3 import NSGA3
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM
from pymoo.operators.sampling.rnd import FloatRandomSampling
from pymoo.operators.selection.tournament import TournamentSelection, compare
from pymoo.util.ref_dirs import get_reference_directions
from pymoo.core.termination import TerminateIfAny
from pymoo.termination import get_termination

from argparser import minimize_C

''' 
OPTIMIZATION PARAMETERS:

   1) Time window (which is set in Economic_parameters
   2) State of Charge (SoC) Initialization
   3) Population
   4) n_var = number of variables to be optimized to minimize/maximize the objective function
   5) n_obj = number of objects to be minimized/maximized
   6) xl = lower bound constraints for the n_var variables
   7) xu = upper bound constraints for the n_var variables
   8) n_gen number of generations of genes created or Tolerance and number of periods
   9) Termination
   10) Reference Distances
   11) Algorithm and hyperparameters
   12) Plot (boolean value to enable plots)

'''
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

# DEFINE OPTIMIZATION PARAMETERS

# 1) Time Windos

time_window = time_window

# 2) State of Charge Initialization

soc_0 = soc  # Define initial SoC

# 3) Population

pop_size = 100  # Define the population size, which is the number of genes of the NSGA-III

# 4) n_var
if minimize_C:
    n_var = time_window*2
else:
    n_var = time_window

# 5) n_obj

if minimize_C:
    n_obj = 2
else:
    n_obj = 1

# 6) xl

if minimize_C:
    xl = [-1] * (time_window) + [0.2] * (time_window)
else:
    xl = [-1]*time_window

# 7) xu

if minimize_C:
    xu = [1] * (time_window) + [1] *  (time_window)
else:
    xu = [1]*time_window

# 8) n_gen

n_gen = 2000

# 8-bis) Tolerance and period number

if minimize_C:
    tolerance = 0.1
    period = 20
else:
    tolerance = 0.5  # tolerance on the objective function
    period = 5  # number of iteration in which tolerance is evaluated (0.3 e 20 valori giusti)

# 9) Termination

termination1 = get_termination("n_gen", n_gen)  # replaced by tolerance
termination2 = RobustTermination(DesignSpaceTermination(tol=tolerance), period=period)

termination = TerminateIfAny(termination1, termination2)

# 10) Reference Directions

'''

Most studies have used the Das and Dennis’s structured approach for generating well-spaced reference points.
A reference direction is constructed by a vector originating from the origin and connected to each of them. The number
of points on the unit simplex is determined by a parameter p (we call it n_partitions in our implementation), which
indicates the number of gaps between two consecutive points along an objective axis.

'''

ref_dirs = get_reference_directions("das-dennis", n_obj, n_partitions=20)

# 11) Algorithm: Sampling, Selection, Crossover, Mutation

algorithm = NSGA3(

    pop_size=pop_size,
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
    # The distribution index for SBX. A higher value of eta results in offspring closer to their
    # parents, while a lower value results in more variation.
    # The probability of crossover being applied. A probability of 1.0 means crossover is always
    # applied.

    crossover=SBX(eta=1, prob=1),

    # mutation: This parameter specifies the mutation operator used for generating variation in offspring.
    # PM: Polynomial Mutation (PM) is a common mutation method for real-valued variables.
    # The distribution index for PM. Similar to SBX, a higher value of eta results in smaller mutations,
    # while a lower value results in larger mutations.

    mutation=PM(eta=80),

    eliminate_duplicates=True

)

# 12 Plots, boolean value to enable or not plots

plot = True


