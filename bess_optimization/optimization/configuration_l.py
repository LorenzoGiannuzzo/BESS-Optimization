""" BESS Optimization using Various Genetic Algorithms

    __author__ = "Lorenzo Giannuzzo"
    __maintainer__ = "Lorenzo Giannuzzo"
    __email__ = "lorenzo.giannuzzo@polito.it"
    __status__ = "in progress"
    __version__ = "v0.2.1"
    __license__ = "MIT"

Last Update of current code: 10/04/2025 """

# IMPORT LIBRARIES AND MODULES FROM PROJECT FILES
import numpy as np
from argparser_l import soc
from Economic_parameters_l import time_window, season
from pymoo.termination.robust import RobustTermination
from pymoo.termination.xtol import DesignSpaceTermination
from pymoo.util.reference_direction import UniformReferenceDirectionFactory, MultiLayerReferenceDirectionFactory
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.algorithms.moo.nsga3 import NSGA3
from pymoo.algorithms.moo.spea2 import SPEA2
from pymoo.algorithms.moo.unsga3 import UNSGA3
from pymoo.algorithms.moo.rnsga3 import RNSGA3  # Import R-NSGA-II
from pymoo.algorithms.moo.moead import MOEAD  # Import MOEA/D
from pymoo.algorithms.soo.nonconvex.brkga import BRKGA
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM
from pymoo.operators.sampling.lhs import LatinHypercubeSampling
from pymoo.operators.selection.tournament import TournamentSelection, compare
from pymoo.util.ref_dirs import get_reference_directions

from pymoo.core.termination import TerminateIfAny
from pymoo.termination import get_termination
from BESS_model_l import charge_rate_interpolated_func, discharge_rate_interpolated_func
from pymoo.config import Config
from utils import CustomSampling
from Economic_parameters_l import PUN_timeseries_buy
from pymoo.operators.survival.rank_and_crowding import RankAndCrowding
Config.warnings['not_compiled'] = False

# IDENTIFICATION OF MAX CHARGE AND DISCHARGE BESS CAPABILITY
x = np.linspace(0, 1, 1000)
charge_values = charge_rate_interpolated_func(x)
discharge_values = discharge_rate_interpolated_func(x)
max_charge = max(charge_values)
max_discharge = max(discharge_values)

# DEFINE RANDOM SET FUNCTION EXTRACTION


def comp_by_cv_then_random(pop, P, **kwargs):
    S = np.full(P.shape[0], np.nan)
    for i in range(P.shape[0]):
        a, b = P[i, 0], P[i, 1]
        if pop[a].CV > 0.0 or pop[b].CV > 0.0:
            S[i] = compare(a, pop[a].CV, b, pop[b].CV, method='smaller_is_better', return_random_if_equal=True)
        else:
            S[i] = np.random.choice([a, b])
    return S[:, None].astype(int)


def binary_tournament(pop, P, **kwargs):
    # The P input defines the tournaments and competitors
    n_tournaments, n_competitors = P.shape

    if n_competitors != 2:
        raise Exception("Only pressure=2 allowed for binary tournament!")

    # the result this function returns
    import numpy as np
    S = np.full(n_tournaments, -1, dtype=np.integer)

    # now do all the tournaments
    for i in range(n_tournaments):
        a, b = P[i]

        # if the first individual is better, choose it
        if pop[a].F < pop[b].F:
            S[i] = a

        # otherwise take the other individual
        else:
            S[i] = b

    return S

# 1) DEFINE TIME WINDOW OBTAINED FROM Economic_parameters.py FILE
time_window = time_window

# 2) SoC at timestep 0 INITIALIZATION DEFINED FROM ARGPARSER
soc_0 = soc

# 3) DEFINE POPULATION SIZE USED TO EXPLORE THE OPTIMIZATION DOMAIN
pop_size = 100

# 4) DEFINE NUMBER OF ELEMENTS INIZIALIZED BY THE ALGORITHM
n_var = time_window * 2

# 5) DEFINE NUMBER OF OBJECTIVES
n_obj = 2

# 6) DEFINE LOWER BOUNDARIES
xl = [-max_discharge] * time_window + [0.0] * time_window

# 7) DEFINE UPPER BOUNDARIES
xu = [max_charge] * time_window + [+1.0] * time_window

# 8) DEFINE NUMBER OF GENERATIONS
n_gen = 200

# 8-bis) DEFINE TOLERANCE
tolerance = 0.05
period = 200
seed = 42

# 9) DEFINITION OF THE TERMINATION CRITERIA
termination1 = get_termination("n_gen", n_gen)
termination2 = RobustTermination(DesignSpaceTermination(tol=tolerance), period=period)
termination = TerminateIfAny(termination1, termination2)

# 10) DEFINITION OF THE REFERENCE DIRECTION
ref_dirs = get_reference_directions("das-dennis", n_obj, n_partitions=12)
ref_dire = UniformReferenceDirectionFactory(n_obj, n_points=pop_size).do()

# DEFINE ETA FOR CROSSOVER
eta_crossover = 20
eta_mutation = 30

# DEFINE ETA FOR MUTATION
prob_crossover = 0.9
prob_mutation = 1.0
n_offsprings = 50

season = season

# 11) ALGORITHM SELECTION
algorithm_type = "NSGA3"


# ALGORITHM INITIALIZATION
if algorithm_type == "NSGA3":
    algorithm = NSGA3(
        pop_size=pop_size,
        ref_dirs=ref_dirs,
        sampling=LatinHypercubeSampling(),
        crossover=SBX(eta=eta_crossover, prob=prob_crossover),
        mutation=PM(eta=eta_mutation, prob=prob_mutation),
        eliminate_duplicates=True,
        seed=seed,
        n_offsprings=n_offsprings
    )
elif algorithm_type == "NSGA_2":
    algorithm = NSGA2(
        pop_size=pop_size,
        sampling=LatinHypercubeSampling(),
        crossover=SBX(eta=eta_crossover, prob=prob_crossover),
        mutation=PM(eta=eta_mutation, prob=prob_mutation),
        eliminate_duplicates=True,
        seed=seed,
        n_offsprings=n_offsprings
    )

elif algorithm_type == "SPEA2":
    algorithm = SPEA2(
        pop_size=pop_size,
        sampling=LatinHypercubeSampling(),
        crossover=SBX(eta=eta_crossover, prob=prob_crossover),
        mutation=PM(eta=eta_mutation, prob=prob_mutation),
        eliminate_duplicates=True,
        seed=seed,
        n_offsprings=n_offsprings
    )
elif algorithm_type == "UNSGA3":
    algorithm = UNSGA3(
        pop_size=pop_size,
        ref_dirs=ref_dirs,
        sampling=LatinHypercubeSampling(),
        crossover=SBX(eta=eta_crossover, prob=prob_crossover),
        mutation=PM(eta=eta_mutation, prob=prob_mutation),
        eliminate_duplicates=True,
        seed=seed,
        n_offsprings = n_offsprings,


    )

elif algorithm_type == "MOEAD":
    algorithm = MOEAD(
        n_neighbors=5,
        prob_neighbor_mating=0.9,
        ref_dirs= get_reference_directions("das-dennis", n_obj, n_partitions=n_obj),
        sampling=LatinHypercubeSampling(),
        crossover=SBX(eta=eta_crossover, prob=prob_crossover),
        mutation=PM(eta=eta_mutation, prob=prob_mutation),
        eliminate_duplicates=True,
        seed=seed,
        n_offsprings=n_offsprings
    )

elif algorithm_type == "BRKGA":
    algorithm = BRKGA(
        n_elites=10,
        sampling=LatinHypercubeSampling(),
        eliminate_duplicates=True,
        seed=seed,
        n_offsprings=n_offsprings,
        bias=0.5,  # You can adjust the bias parameter as needed
    )

elif algorithm_type == "RNSGA3":
    algorithm = RNSGA3(
        sampling=LatinHypercubeSampling(),
        crossover=SBX(eta=eta_crossover, prob=prob_crossover),
        mutation=PM(eta=eta_mutation, prob=prob_mutation),
        eliminate_duplicates=True,
        seed=seed,
        mu=0.0035,
        n_offsprings=n_offsprings,
        ref_points=ref_dire,
        pop_per_ref_point=2
    )

else:
    raise ValueError(f"Algorithm '{algorithm_type}' is not recognized.")

# 12 DEFINE BOOLEAN VALUE TO ENABLE/DISABLE PLOTS
plot = True
plot_monthly = False


i = 0
