import numpy as np
from pymoo.core.problem import ElementwiseProblem
from BESS_parameters import size, file_path, file_path2, sheetname, sheetname2, sheetname3
from Economic_parameters import PUN_timeseries
from BESS_parameters import charge_rate_interpolated_func, discharge_rate_interpolated_func
from pymoo.termination import get_termination


''' 
OPTIMIZATION PARAMETERS:

   1) Time window
   2) State of Charge (SoC) Initialization
   3) Population

'''

# DEFINE OPTIMIZATION PARAMETERS

time_window = 48  # Define time window optimization
soc_0 = 0.2  # Define initial SoC in %
pop_size = 100 # Define the population size, which is the number of genes of the NSGA-III
termination = get_termination("n_gen",100)


# DEFINE OPTIMIZATION PROBLEM
class Revenues(ElementwiseProblem):
    def __init__(
            self,
            size,
            pop_size,
            file_path2,
            sheetname3,**kwargs

    ) -> None:
        super().__init__(
            n_var=time_window,
            n_obj=1,
            xl=[-1] * time_window,
            xu=[1] * time_window,
            vtype=float, **kwargs
        )

        # DEFINE REVENUES ATTRIBUTES FROM IMPORTER PARAMETERS

        self.PUN_timeseries = PUN_timeseries
        self.c_func, self.d_func = charge_rate_interpolated_func, discharge_rate_interpolated_func

        # DEFINE OBJECTIVE FUNCTION PARAMETERS

        self.soc = np.zeros((len(self.PUN_timeseries)))
        self.charged_energy = np.zeros((len(self.PUN_timeseries)))
        self.discharged_energy = np.zeros((len(self.PUN_timeseries)))
        self.c_func, self.d_func = charge_rate_interpolated_func, discharge_rate_interpolated_func

        # INITIALIZE SoC AT t=0

        self.soc[0] = soc_0

    # OBJECTIVE FUNCTION DEFINITION

    def _evaluate(self, x, out, *args, **kwargs):

        # SET X-VECTOR TO BE OPTIMIZED AS THE % OF CHARGED AND DISCHARGED ENERGY FROM BESS

        self.c_d_timeseries = np.array(x).reshape(time_window)

        # EVALUATE THE CHARGED AND DISCHARGED ENERGY AND UPDATE THE SoC FOR EACH TIMESTEP t

        for index in range(len(self.PUN_timeseries) - 1):

            # APPLY PHYSICAL CONSTRAINTS TO BESS DURING THE TIMESTEP t

            if self.c_d_timeseries[index] >= 0.0:
                self.c_d_timeseries[index] = np.minimum(self.c_d_timeseries[index],
                                                 np.minimum(self.c_func(self.soc[index]), 0.9 - self.soc[index]))
            else:
                self.c_d_timeseries[index] = np.maximum(self.c_d_timeseries[index],
                                                 np.maximum(-self.d_func(self.soc[index]), 0.1 - self.soc[index]))

            # EVALUATE CHARGED AND DISCHARGED ENERGY FOR THE TIMESTEP t

            if self.c_d_timeseries[index] >= 0:
                self.charged_energy[index] = self.c_d_timeseries[index] * size
                self.discharged_energy[index] = 0.0
            elif self.c_d_timeseries[index] <= 0:
                self.discharged_energy[index] = self.c_d_timeseries[index] * size
                self.charged_energy[index] = 0.0
            else:
                self.charged_energy[index] = 0.0
                self.discharged_energy[index] = 0.0

            # UPDATE THE SoC FOR THE TIMESTEP t

            if self.c_d_timeseries[index] >= 0.0:
                self.soc[index + 1] = np.minimum(1, self.soc[index] + self.charged_energy[index] / size)
            else:
                self.soc[index + 1] = max(0.0, self.soc[index] + self.discharged_energy[index] / size)

        # EVALUATE THE REVENUES OBTAINED FOR EACH TIMESTEP t

        revenue_column = np.array(-(self.discharged_energy * self.PUN_timeseries / 1000) - (self.charged_energy * self.PUN_timeseries / 1000))

        # EVALUATE THE REVENUES OBTAINED DURING THE OPTIMIZATION TIME WINDOW

        total_revenue = sum(revenue_column)

        # CORRECT THE VALUES OF THE REVENUES IN ORDER TO MINIMIZE THE OBJECTIVE FUNCTION

        final_revenues = -total_revenue

        # DEFINE THE OUTPUT OF THE OPTIMIZATION PROBLEM

        out["F"] = final_revenues
