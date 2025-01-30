"""

BESS Optimization using NSGA-III Algorithm

    __author__ = "Lorenzo Giannuzzo"
    __maintainer__ = "Lorenzo Giannuzzo"
    __email__ = "lorenzo.giannuzzo@polito.it"
    __status__ = "in progress"
    __version__ = "v0.2.1"
    __license__ = "MIT"

Last Update of current code: 09/01/2025 - 12:20

"""

import numpy as np
import configuration_l
import Economic_parameters_l
from pymoo.core.problem import ElementwiseProblem
from BESS_model_l import BESS_model, charge_rate_interpolated_func, discharge_rate_interpolated_func, size
from PV_l import pv_production

# DEFINE OPTIMIZATION PROBLEM
class Revenues(ElementwiseProblem):
    def __init__(
            self,
            **kwargs

    ) -> None:
        super().__init__(
            n_var= configuration_l.n_var,
            n_obj=configuration_l.n_obj,
            xl= configuration_l.xl,
            xu= configuration_l.xu,
            vtype=float,
            **kwargs,

        )

        # DEFINE REVENUES ATTRIBUTES FROM IMPORTER PARAMETERS
        self.PUN_timeseries = Economic_parameters_l.PUN_timeseries[:, 1]
        self.c_func, self.d_func = charge_rate_interpolated_func, discharge_rate_interpolated_func

        # DEFINE OBJECTIVE FUNCTION PARAMETERS
        self.soc = np.zeros((len(self.PUN_timeseries)))
        self.charged_energy = np.zeros((len(self.PUN_timeseries)))
        self.discharged_energy = np.zeros((len(self.PUN_timeseries)))

        # INITIALIZE SoC AT t=0
        self.soc[0] = configuration_l.soc_0
        self.time_window = configuration_l.time_window
        self.size = size
        self.production = pv_production['P']

    # OBJECTIVE FUNCTION DEFINITION
    def _evaluate(self, x, out, *args, **kwargs):

        # SET X-VECTOR TO BE OPTIMIZED AS THE % OF CHARGED AND DISCHARGED ENERGY FROM BESS
        self.c_d_timeseries = np.array(x[:self.time_window]).reshape(configuration_l.time_window)
        self.alpha = np.ones(self.time_window)

        # EVALUATE THE CHARGED AND DISCHARGED ENERGY AND UPDATE THE SoC FOR EACH TIMESTEP t
        # Create an instance of BESS_model
        bess_model = BESS_model(self.time_window, self.PUN_timeseries, self.soc, self.size, self.c_func,
                                    self.d_func)
        # Run the simulation
        self.charged_energy, self.discharged_energy = bess_model.run_simulation(self.c_d_timeseries)
        self.taken_from_pv = np.minimum(self.charged_energy, self.production)
        self.charged_energy_grid = np.maximum(self.charged_energy - self.taken_from_pv, 0.0)
        self.discharged_from_pv = np.minimum(-self.production + self.taken_from_pv, 0.0)

        # APPLY POD CONSTRAINTS
        from argparser_l import n_cycles

        for i in range(len(self.discharged_from_pv)):

            from argparser_l import POD_power

            if -self.discharged_from_pv[i] - self.discharged_energy[i] > POD_power:

                self.discharged_from_pv[i] = -min(POD_power, -self.discharged_from_pv[i])

                self.discharged_energy[i] = -min(POD_power - abs(self.discharged_from_pv[i]),
                                                     -self.discharged_energy[i])
            if self.charged_energy_grid[i] >= POD_power:

                self.charged_energy_grid[i] = min(self.charged_energy_grid[i], POD_power)
                self.charged_energy[i] = self.charged_energy_grid[i] + self.taken_from_pv[i]

        for i in range(self.time_window - 1):

            # EVALUATE SOC MAX
            from BESS_model_l import degradation
            from argparser_l import soc_max, soc_min
            n_cycles_prev = n_cycles
            max_capacity = degradation(n_cycles_prev) / 100
            soc_max = min(soc_max, max_capacity)

            # Update SoC for the next time step
            if self.c_d_timeseries[i] >= 0:

                self.soc[i + 1] = min(soc_max, self.soc[i] + self.charged_energy[i] / size)
                self.charged_energy[i] = (self.soc[i + 1] - self.soc[i]) * size
            else:
                self.soc[i + 1] = max(soc_min, self.soc[i] + self.discharged_energy[i] / size)
                self.discharged_energy[i] = (self.soc[i + 1] - self.soc[i]) * size

            total_energy = self.charged_energy[i] + np.abs(self.discharged_energy[i])
            actual_capacity = size * degradation(n_cycles_prev) / 100
            n_cycles = n_cycles_prev + total_energy / actual_capacity

        # EVALUATE THE NUMBER OF CYCLES DONE BY BESS
        total_charged = np.sum(self.charged_energy)
        total_discharged = np.sum(-self.discharged_energy)
        total_energy = total_charged + total_discharged

        from argparser_l import n_cycles

        n_cycles_prev = n_cycles
        actual_capacity = size * degradation(n_cycles_prev)/100
        n_cycles = total_energy / actual_capacity

        # EVALUATE THE REVENUES OBTAINED FOR EACH TIMESTEP t
        revenue_column = np.array(-(self.discharged_energy * self.PUN_timeseries / 1000) -
                                      (self.charged_energy_grid * self.PUN_timeseries / 1000)
                                      - (self.discharged_from_pv * self.PUN_timeseries / 1000))

        # EVALUATE THE REVENUES OBTAINED DURING THE OPTIMIZATION TIME WINDOW
        total_revenue = sum(revenue_column)

        # CORRECT THE VALUES OF THE REVENUES IN ORDER TO MINIMIZE THE OBJECTIVE FUNCTION
        final_revenues = -total_revenue

        # DEFINE THE OUTPUT OF THE OPTIMIZATION PROBLEM
        out["F"] = [final_revenues]



