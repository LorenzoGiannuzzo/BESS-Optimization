""" BESS Optimization using NSGA-III Algorithm

    __author__ = "Lorenzo Giannuzzo"
    __maintainer__ = "Lorenzo Giannuzzo"
    __email__ = "lorenzo.giannuzzo@polito.it"
    __status__ = "in progress"
    __version__ = "v0.2.1"
    __license__ = "MIT"

Last Update of current code: 19/02/2025 """

# IMPORT LIBRARIES AND MODULES
import numpy as np
import configuration_s
import Economic_parameters_s
from pymoo.core.problem import ElementwiseProblem
from BESS_model_s import BESS_model, charge_rate_interpolated_func, discharge_rate_interpolated_func, size
from PV_s import pv_production
from Load import data
from argparser_s import POD_power
from BESS_model_s import power_energy
from BESS_model_s import degradation


# DEFINE OPTIMIZATION PROBLEM
class Revenues(ElementwiseProblem):
    def __init__(
            self,
            **kwargs
    ) -> None:
        super().__init__(
            n_var=configuration_s.n_var,
            n_obj=configuration_s.n_obj,
            xl=configuration_s.xl,
            xu=configuration_s.xu,
            vtype=float,
            **kwargs,
        )

        # DEFINE REVENUES ATTRIBUTES FROM IMPORTER PARAMETERS - PUN and C/D Functions
        self.PUN_timeseries = Economic_parameters_s.PUN_timeseries[:, 1]
        self.c_func, self.d_func = charge_rate_interpolated_func, discharge_rate_interpolated_func

        # DEFINE OBJECTIVE FUNCTION PARAMETERS - SoC, Charged/Discharged Energy (from BESS)
        self.soc = np.zeros((len(self.PUN_timeseries)))
        self.charged_energy_to_BESS = np.zeros((len(self.PUN_timeseries)))
        self.discharged_energy_from_BESS = np.zeros((len(self.PUN_timeseries)))
        # self.load_self_consumption = np.zeros((len(self.PUN_timeseries)))
        # self.from_pv_to_load = np.zeros((len(self.PUN_timeseries)))
        # self.from_BESS_to_load = np.zeros((len(self.PUN_timeseries)))
        # self.load_decision = np.zeros((len(self.PUN_timeseries)))
        self.shared_energy_REC = np.zeros((len(self.PUN_timeseries)))
        self.shared_energy_BESS = np.zeros((len(self.PUN_timeseries)))
        # self.taken_from_pv = np.zeros((len(self.PUN_timeseries)))
        self.charged_energy_from_grid_to_BESS = np.zeros((len(self.PUN_timeseries)))
        self.discharged_from_pv = np.zeros((len(self.PUN_timeseries)))

        # INITIALIZE SoC AT t=0
        self.soc[0] = configuration_s.soc_0

        # DEFINE TIME WINDOW AND BESS SIZE
        self.time_window = configuration_s.time_window
        self.size = size

        # DEFINE PV PRODUCTION AND LOAD CONSUMTPION
        self.production = pv_production['P']
        self.rec_load = data

        self.remaining_production = np.zeros((len(self.PUN_timeseries)))

    # OBJECTIVE FUNCTION DEFINITION
    def _evaluate(self, x, out, *args, **kwargs):

        # SET X-VECTOR TO BE OPTIMIZED AS THE % OF CHARGED AND DISCHARGED ENERGY FROM BESS
        self.c_d_timeseries = np.array(x[:self.time_window]).reshape(configuration_s.time_window)

        # EVALUATE THE CHARGED AND DISCHARGED ENERGY AND UPDATE THE SoC FOR EACH TIMESTEP t
        # Create an instance of BESS_model
        bess_model = BESS_model(self.time_window, self.PUN_timeseries, self.soc, self.size, self.c_func,
                                    self.d_func)

        # GET CHARGED/DISCHARGED VALUES FROM BESS MODEL RUN
        self.charged_energy_to_BESS, self.discharged_energy_from_BESS = bess_model.run_simulation(self.c_d_timeseries)

        # EXECUTE THE UPDATE FOR EACH i-th TIMESTEP OF ALL THE ENERGY VECTORS. EVALUATING ENERGY BALANCES
        from argparser_s import n_cycles
        from argparser_s import soc_max, soc_min

        for i in range(self.time_window - 1):

            # UPDATE SOC MAX BASED ON ITS ACTUAL AND PAST DEGRADATION
            n_cycles_prev = n_cycles
            max_capacity = degradation(n_cycles_prev) / 100
            soc_max = min(soc_max, max_capacity)

            # EVALUATE THE ENERGY USED TO CHARGE THE BESS TAKEN FROM THE GRID (IF CHARGED_ENERGY_FROM_BESS IS NEGATIVE,
            # MEANING THAT THE BESS IS DISCHARGING, THIS VALUE IS = 0
            self.charged_energy_from_grid_to_BESS[i] = self.charged_energy_to_BESS[i]

            if np.sign(self.charged_energy_from_grid_to_BESS[i]) < 0:
                print("Warning: Charged Energy from Grid to BESS is Negative!\n\n")

            # UPDATE THE ENERGY DISCHARGED FROM PV DIRECTLY TO THE GRID REDUCING ITS ORIGINAL VALUE BY THE ONE THAT
            # GOES FROM PV TO THE BESS AND FROM THE PV TO THE LOAD
            self.discharged_from_pv[i] = -self.production[i]  # NEGATIVE VALUE

            if np.sign(self.discharged_from_pv[i]) > 0:
                print("Warning: Discharged Energy from PV is Positive!\n\n")

            # EVALUATE THE ENERGY DISCHARGED FROM BESS BASED TO THE VALUE OF THE OTHER OBTAINED ENERGY VECTORS
            self.discharged_energy_from_BESS[i] = -np.minimum(np.minimum(np.abs(self.discharged_energy_from_BESS[i]),
                                                             (self.soc[i]-soc_min)*size*power_energy),
                                                             size*power_energy)  # TODO: REMOVE REDUNDANCY

            if np.sign(self.discharged_energy_from_BESS[i]) > 0:
                print("Warning: Discharged Energy from BESS is Positive!\n\n")

            # APPLY POD CONSTRAINTS TO ENERGY VECTORS

            # IF POD POWER IS EXCEEDED WHILE TAKING ENERGY FROM THE GRID
            if self.charged_energy_from_grid_to_BESS[i] > POD_power:

                # THEN LIMIT ALSO THE ENERGY CHARGED FROM GRID TO BESS (CONTROLLABLE)
                self.charged_energy_from_grid_to_BESS[i] = np.minimum(POD_power, self.charged_energy_from_grid_to_BESS[i])

                if np.sign(self.charged_energy_from_grid_to_BESS[i]) < 0:
                    print("Warning: Charged Energy from Grid to BESS is Negative when updating POD Constraints!\n\n")

            # IF POD POWER IS EXCEEDED WHILE DISCHARGING ENERGY TO THE GRID
            if np.abs(self.discharged_energy_from_BESS[i]) > POD_power:

                # THEN ALSO LIMIT THE ENERGY DISCHARGED FROM BESS TO THE GRID (CONTROLLABLE)
                self.discharged_energy_from_BESS[i] = -np.minimum(POD_power, np.abs(self.discharged_energy_from_BESS[i]))

                if np.sign(self.discharged_energy_from_BESS[i]) > 0:
                    print("Warning: Discharged Energy from BESS is Positive when updating POD Constraints!\n\n")

            # UPDATE AGAIN ALL THE ENERGY VECTOR BASED ON POD POWER CONSTRAINS THAT WERE APPLIED

            # UPDATE THE ENERGY THAT THE BESS WANT TO BE CHARGED AS SUM OF THE ONE CHARGED FROM GRID TO BESS AND THE
            # ENERGY TAKEN FROM PV TO THE BESS
            self.charged_energy_to_BESS[i] = self.charged_energy_from_grid_to_BESS[i]

            if np.sign(self.charged_energy_to_BESS[i]) < 0:
                print("Warning: Charged Energy from grid to BESS is Negative!\n\n")

            # UPDATE THE ENERGY DISCHARGED FROM PV DIRECTLY TO THE GRID REDUCING ITS ORIGINAL VALUE BY THE ONE THAT
            self.discharged_energy_from_BESS[i] = -np.minimum(np.minimum(np.abs(self.discharged_energy_from_BESS[i]),
                                                             (self.soc[i]-soc_min)*size*power_energy),
                                                             size*power_energy)

            if np.sign(self.discharged_energy_from_BESS[i]) > 0:
                print("Warning: Discharged Energy from BESS is Positive!\n\n")

            # UPDATE SOC

            # IF BESS IS CHARGING
            if self.charged_energy_from_grid_to_BESS[i] > 0:
                self.soc[i + 1] = min(soc_max, self.soc[i] + (self.charged_energy_to_BESS[i]
                                                            ) / size)
            # IF BESS IS DISCHARGING
            elif self.discharged_energy_from_BESS[i] < 0:
                self.soc[i + 1] = max(soc_min, self.soc[i] - (np.abs(self.discharged_energy_from_BESS[i])
                                                              ) / size)
            else:
                self.soc[i+1] = self.soc[i]


            # EVALUATING SHARED ENERGY
            self.shared_energy_REC[i] = np.minimum(self.rec_load[i], np.abs(self.discharged_from_pv[i]))
            self.remaining_production[i] = np.maximum(np.abs(self.discharged_from_pv[i]) - self.shared_energy_REC[i], 0.0)
            self.shared_energy_BESS[i] = np.minimum(self.remaining_production[i], self.charged_energy_to_BESS[i])

            total_energy = self.charged_energy_to_BESS[i] + np.abs(self.discharged_energy_from_BESS[i])
            actual_capacity = size * degradation(n_cycles_prev) / 100
            n_cycles = n_cycles_prev + total_energy / actual_capacity

        # EVALUATE THE NUMBER OF CYCLES DONE BY BESS
        total_charged = np.sum(self.charged_energy_to_BESS)
        total_discharged = np.sum(-self.discharged_energy_from_BESS)
        total_energy = total_charged + total_discharged

        from argparser_s import n_cycles

        n_cycles_prev = n_cycles
        actual_capacity = size * degradation(n_cycles_prev)/100

        n_cycles = total_energy / actual_capacity

        # EVALUATE THE REVENUES OBTAINED FOR EACH TIMESTEP t
        revenue_column = np.array(np.abs(self.discharged_energy_from_BESS) * self.PUN_timeseries / 1000 -
                                      np.abs(self.charged_energy_from_grid_to_BESS) * self.PUN_timeseries / 1000
                                      # + self.discharged_from_pv * self.PUN_timeseries / 1000
                                      + np.abs(self.shared_energy_BESS) * 120 / 1000
                                  )

        #print(self.PUN_timeseries/1000)

        # EVALUATE REVENUES CONSIDERING TYPICAL DAYS FOR EACH MONTH
        # num_settimane = 12
        # ore_per_settimana = 24
        # revenues_settimanali = np.zeros(num_settimane)
        #
        # for i in range(num_settimane):
        #     inizio = i * ore_per_settimana
        #     fine = inizio + ore_per_settimana
        #     revenues_settimanali[i] = np.sum(revenue_column[inizio:fine]) * 30

        revenues_settimanali = np.sum(revenue_column)
        revenues_finali = revenues_settimanali
        somma_revenues_finali = np.sum(revenues_finali)

        # EVALUATE THE REVENUES OBTAINED DURING THE OPTIMIZATION TIME WINDOW
        total_revenue = somma_revenues_finali

        # CORRECT THE VALUES OF THE REVENUES IN ORDER TO MINIMIZE THE OBJECTIVE FUNCTION
        final_revenues = -total_revenue

        # DEFINE THE OUTPUT OF THE OPTIMIZATION PROBLEM
        out["F"] = [final_revenues]



