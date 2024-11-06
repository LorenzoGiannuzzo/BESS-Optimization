import numpy as np
import configuration
import Economic_parameters

from pymoo.core.problem import ElementwiseProblem
from BESS_model import BESS_model, charge_rate_interpolated_func, discharge_rate_interpolated_func, size
from argparser import minimize_C
from PV import pv_production

# DEFINE OPTIMIZATION PROBLEM

class Revenues(ElementwiseProblem):
    def __init__(
            self,
            **kwargs

    ) -> None:
        super().__init__(
            n_var= configuration.n_var,
            n_obj=configuration.n_obj,
            xl= configuration.xl,
            xu= configuration.xu,
            vtype=float,
            **kwargs,

        )

        # DEFINE REVENUES ATTRIBUTES FROM IMPORTER PARAMETERS

        self.PUN_timeseries = Economic_parameters.PUN_timeseries[:,1]
        self.c_func, self.d_func = charge_rate_interpolated_func, discharge_rate_interpolated_func

        # DEFINE OBJECTIVE FUNCTION PARAMETERS

        self.soc = np.zeros((len(self.PUN_timeseries)))
        self.charged_energy = np.zeros((len(self.PUN_timeseries)))
        self.discharged_energy = np.zeros((len(self.PUN_timeseries)))

        # INITIALIZE SoC AT t=0

        self.soc[0] = configuration.soc_0

        self.time_window = configuration.time_window
        self.size = size

        self.production = pv_production['P']

    # OBJECTIVE FUNCTION DEFINITION

    def _evaluate(self, x, out, *args, **kwargs):

        # SET X-VECTOR TO BE OPTIMIZED AS THE % OF CHARGED AND DISCHARGED ENERGY FROM BESS
        if minimize_C:
            self.c_d_timeseries = np.array(x[:self.time_window]).reshape(configuration.time_window)
            self.alpha = np.array(x[self.time_window:self.time_window*2])

            # EVALUATE THE CHARGED AND DISCHARGED ENERGY AND UPDATE THE SoC FOR EACH TIMESTEP t

            # Create an instance of BESS_model

            bess_model = BESS_model(self.time_window, self.PUN_timeseries, self.soc, self.size, self.c_func, self.d_func, self.alpha)

            # Run the simulation

            self.charged_energy, self.discharged_energy = bess_model.run_simulation(self.c_d_timeseries, self.alpha)

            self.taken_from_pv = np.minimum(self.charged_energy, self.production)

            self.charged_energy_grid = np.maximum(self.charged_energy - self.taken_from_pv, 0.0)

            #self.charged_energy = np.maximum(self.charged_energy - self.production, 0.0)

            self.discharged_from_pv = np.minimum(-self.production + self.taken_from_pv, 0.0)


            # EVALUATE THE REVENUES OBTAINED FOR EACH TIMESTEP t

            revenue_column = np.array(
                -(self.discharged_energy * self.PUN_timeseries / 1000) - (self.charged_energy_grid *
                                                                          self.PUN_timeseries / 1000) - (
                            self.discharged_from_pv * self.PUN_timeseries / 1000))

            # EVALUATE THE REVENUES OBTAINED DURING THE OPTIMIZATION TIME WINDOW

            total_revenue = sum(revenue_column)

            # CORRECT THE VALUES OF THE REVENUES IN ORDER TO MINIMIZE THE OBJECTIVE FUNCTION

            final_revenues = -total_revenue
            alpha = np.sum(self.alpha)

            # DEFINE THE OUTPUT OF THE OPTIMIZATION PROBLEM

            out["F"] = [final_revenues, alpha]

        else:
            self.c_d_timeseries = np.array(x[:self.time_window]).reshape(configuration.time_window)
            self.alpha = np.ones(self.time_window)

            # EVALUATE THE CHARGED AND DISCHARGED ENERGY AND UPDATE THE SoC FOR EACH TIMESTEP t

            # Create an instance of BESS_model

            bess_model = BESS_model(self.time_window, self.PUN_timeseries, self.soc, self.size, self.c_func,
                                    self.d_func, self.alpha)

            # Run the simulation

            self.charged_energy, self.discharged_energy = bess_model.run_simulation(self.c_d_timeseries, self.alpha)

            self.taken_from_pv = np.minimum(self.charged_energy, self.production)

            self.charged_energy_grid = np.maximum(self.charged_energy - self.taken_from_pv, 0.0)

            #self.charged_energy = np.maximum(self.charged_energy - self.production, 0.0)

            self.discharged_from_pv = np.minimum(-self.production + self.taken_from_pv, 0.0)


            #print(sum(self.discharged_energy))


            # EVALUATE THE REVENUES OBTAINED FOR EACH TIMESTEP t

            revenue_column = np.array(-(self.discharged_energy * self.PUN_timeseries / 1000) - (self.charged_energy_grid *
                                                                                              self.PUN_timeseries / 1000)
                                      -(self.discharged_from_pv * self.PUN_timeseries / 1000))
#
            # EVALUATE THE REVENUES OBTAINED DURING THE OPTIMIZATION TIME WINDOW

            total_revenue = sum(revenue_column)


            # CORRECT THE VALUES OF THE REVENUES IN ORDER TO MINIMIZE THE OBJECTIVE FUNCTION

            final_revenues = -total_revenue

            # DEFINE THE OUTPUT OF THE OPTIMIZATION PROBLEM

            out["F"] = [final_revenues]


