"""
BESS Optimization using NSGA-III Algorithm

    __author__ = "Lorenzo Giannuzzo"
    __maintainer__ = "Lorenzo Giannuzzo"
    __email__ = "lorenzo.giannuzzo@polito.it"
    __status__ = "in progress"
    __version__ = "v0.2.1"
    __license__ = "MIT"

Last Update of current code: 17/01/2025 - 12:20
"""
# IMPORT LIBRARIES
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
            n_var= configuration_s.n_var,
            n_obj=configuration_s.n_obj,
            xl= configuration_s.xl,
            xu= configuration_s.xu,
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
        self.load_self_consumption = np.zeros((len(self.PUN_timeseries)))
        self.from_pv_to_load = np.zeros((len(self.PUN_timeseries)))
        self.from_BESS_to_load = np.zeros((len(self.PUN_timeseries)))
        self.load_decision = np.zeros((len(self.PUN_timeseries)))

        # INITIALIZE SoC AT t=0
        self.soc[0] = configuration_s.soc_0

        # DEFINE TIME WINDOW AND BESS SIZE
        self.time_window = configuration_s.time_window
        self.size = size

        # DEFINE PV PRODUCTION AND LOAD CONSUMTPION
        self.production = pv_production['P']
        self.load = data

    # OBJECTIVE FUNCTION DEFINITION
    def _evaluate(self, x, out, *args, **kwargs):

        # SET X-VECTOR TO BE OPTIMIZED AS THE % OF CHARGED AND DISCHARGED ENERGY FROM BESS
        self.c_d_timeseries = np.array(x[:self.time_window]).reshape(configuration_s.time_window)
        self.load_decision = np.array(x[self.time_window:2*self.time_window]).reshape(configuration_s.time_window)

        # EVALUATE THE CHARGED AND DISCHARGED ENERGY AND UPDATE THE SoC FOR EACH TIMESTEP t
        # Create an instance of BESS_model
        bess_model = BESS_model(self.time_window, self.PUN_timeseries, self.soc, self.size, self.c_func,
                                    self.d_func)

        # GET CHARGED/DISCHARGED VALUES FROM BESS MODEL RUN
        self.charged_energy_from_BESS, self.discharged_energy_from_BESS = bess_model.run_simulation(self.c_d_timeseries)

        # INITIALIZE VARIABLES
        total_available_energy = np.zeros((len(self.PUN_timeseries)))
        self.taken_from_pv = np.zeros((len(self.PUN_timeseries)))
        self.charged_energy_from_grid_to_BESS = np.zeros((len(self.PUN_timeseries)))
        self.discharged_from_pv = np.zeros((len(self.PUN_timeseries)))

        # EXECUTE THE UPDATE FOR EACH i-th TIMESTEP OF ALL THE ENERGY VECTORS. EVALUATING ENERGY BALANCES
        from argparser_s import n_cycles
        from argparser_s import soc_max, soc_min

        for i in range(self.time_window - 1):

            # UPDATE SOC MAX BESED ON ITS ACTUAL AND PAST DEGRADATION
            # GET PREVIOUS NUMBER OF CYCLES TODO: THIS COULD BE CHANGED TO SIMPLY THE USER INTERFACE AS INPUT PARAMETERS
            n_cycles_prev = n_cycles
            max_capacity = degradation(n_cycles_prev) / 100
            soc_max = min(soc_max, max_capacity)

            # EVALUATE THE TOTAL AVAILABLE ENERGY TO PERFORM SELF-CONSUMPTION AT THE i-th TIMESTEP
            # WHICH IS EVALUATED AS THE BES SOC ABOVE THE LOWER LIMIT VALUE * BESS_SIZE * P/E_RATIO [kWh] +
            # THE ENERGY PRODUCED FROM PV (Considering also WHAT THE BESS CAN DO, which is BESS_SIZE*P/E_RATIO
            # In other words is the energy that the BESS can give to the load + the energy produced by PV tp perform
            # self-consumption

            total_available_energy[i] = (np.minimum((self.soc[i] - soc_min) * size * power_energy, size * power_energy)
                                         + self.production[i])

            # EVALUATE THE LOAD SELF-CONSUMPTION AS MINIMUM BETWEEN LOAD AND THE ENERGY AVAILABLE
            self.load_self_consumption[i] = self.load_decision[i] * np.minimum(self.load[i], total_available_energy[i])

            # EVALUATE THE ENERGY THAT GOES FROM PV TO LOAD FOR SELF-CONSUMPTION PURPOSES
            self.from_pv_to_load[i] = np.minimum(self.load_self_consumption[i], self.production[i])

            # EVALUATE THE ENERGY TAKEN BY THE BESS FROM PV
            self.taken_from_pv[i] = np.minimum(np.maximum(self.production[i] - self.from_pv_to_load[i], 0.0), self.charged_energy_from_BESS[i])

            # EVALUATE THE ENERGY USED TO CHARGE THE BESS TAKEN FROM THE GRID (IF CHARGED_ENERGY_FROM_BESS IS NEGATIVE,
            # MEANING THAT THE BESS IS DISCHARGING, THIS VALUE IS = 0
            self.charged_energy_from_grid_to_BESS[i] = np.maximum(self.charged_energy_from_BESS[i] - self.taken_from_pv[i],0.0)

            # EVALUATE THE ENERGY THAT GOES FROM THE BESS TO THE LOAD FOR SELF-CONSUMPTION
            self.from_BESS_to_load[i] = np.maximum(self.load_self_consumption[i] - self.from_pv_to_load[i], 0.0)

            # UPDATE THE ENERGY THAT THE BESS WANT TO CHARGED AS SUM OF THE ONE CHARGED FROM GRID TO BESS AND THE ENERGY
            # TAKEN FROM PV TO THE BESS
            self.charged_energy_to_BESS[i] = self.charged_energy_from_grid_to_BESS[i] + self.taken_from_pv[i]

            # UPDATE THE ENERGY DISCHARGED FROM PV DIRECTLY TO THE GRID REDUCING ITS ORIGINAL VALUE BY THE ONE THAT
            # GOES FROM PV TO THE BESS AND FROM THE PV TO THE LOAD
            self.discharged_from_pv[i] = np.minimum(-self.production[i] + self.taken_from_pv[i] +
                                                    self.from_pv_to_load[i], 0.0) # NEGATIVE VALUE

            # EVALUATE THE ENERGY DISCHARGED FROM BESS BASED TO THE VALUE OF THE OTHER OBTAINED ENERGY VECTORS
            self.discharged_energy_from_BESS[i] = np.maximum(np.maximum(self.discharged_energy_from_BESS[i],
                                                             -(self.soc[i]-soc_min)*size*power_energy), -size*power_energy)

            # APPLY POD CONSTRAINTS TO ENERGY VECTORS

            # IF POD POWER IS EXCEEDED WHILE TAKING ENERGY FROM THE GRID
            if self.charged_energy_from_grid_to_BESS[i] + self.load[i] > POD_power:

                # FIRST OF ALL, LIMIT THE LOAD IF EVEN THE LOAD EXCEED THE POD POWER LIMITS (NOT CONTROLLABLE)
                self.load[i] = np.minimum(POD_power, self.load[i])

                # THEN LIMIT ALSO THE ENERGY CHARGED FROM GRID TO BESS (CONTROLLABLE)
                self.charged_energy_from_grid_to_BESS[i] = np.maximum(POD_power - self.load[i], 0.0)

            # IF POD POWER IS EXCEEDED WHILE DISCHARGING ENERGY TO THE GRID
            if -np.abs(self.discharged_from_pv[i]) - np.abs(self.discharged_energy_from_BESS[i]) < -POD_power:

                # FIRST OF ALL LIMIT TGE ENERGY DISCHARGED FROM PV IF IT EXCEED ALONE THE POD POWER LIMIT (NOT
                # CONTROLLABLE)
                self.discharged_from_pv[i] = np.maximum(-POD_power, -self.discharged_from_pv[i])

                # THEN ALSO LIMIT THE ENERGY DISCHARGED FROM BESS TO THE GRID (CONTROLLABLE)
                self.discharged_energy_from_BESS[i] = np.minimum(-(POD_power - self.discharged_from_pv[i]), 0.0)

            # AFTER APPLYING POD CONSTRAINTS, LOAD, CHARGED ENERGY FROM BESS, DISCHARGED FROM PV AND DISCHARGED
            # FROM BESS COULD BE CHANGED
            self.load_self_consumption[i] = self.load_decision[i] * np.minimum(self.load[i], total_available_energy[
                    i])

            # UPDATE AGAIN ALL THE ENERGY VECTOR BASED ON POD POWER CONSTRAINS THAT WERE APPLIED

            self.from_pv_to_load[i] = np.minimum(self.load_self_consumption[i], self.production[i])
            self.taken_from_pv[i] = np.minimum(np.maximum(self.production[i] - self.from_pv_to_load[i], 0.0), self.charged_energy_from_BESS[i])
            self.charged_energy_from_grid_to_BESS[i] = self.charged_energy_from_BESS[i] - self.taken_from_pv[i]

            # EVALUATE THE ENERGY THAT GOES FROM THE BESS TO THE LOAD FOR SELF-CONSUMPTION
            self.from_BESS_to_load[i] = np.maximum(self.load_self_consumption[i] - self.from_pv_to_load[i], 0.0)

            # UPDATE THE ENERGY THAT THE BESS WANT TO CHARGED AS SUM OF THE ONE CHARGED FROM GRID TO BESS AND THE ENERGY
            # TAKEN FROM PV TO THE BESS
            self.charged_energy_to_BESS[i] = self.charged_energy_from_grid_to_BESS[i] + self.taken_from_pv[i]

            # UPDATE THE ENERGY DISCHARGED FROM PV DIRECTLY TO THE GRID REDUCING ITS ORIGINAL VALUE BY THE ONE THAT
            # GOES FROM PV TO THE BESS AND FROM THE PV TO THE LOAD
            self.discharged_from_pv[i] = np.minimum(-self.production[i] + self.taken_from_pv[i] +
                                                        self.from_pv_to_load[i], 0.0)

            self.discharged_energy_from_BESS[i] = np.maximum(np.maximum(self.discharged_energy_from_BESS[i],
                                                             -(self.soc[i]-soc_min)*size*power_energy), -size*power_energy)

            # UPDATE SOC

            # IF BESS I CHARGING
            if self.c_d_timeseries[i] >= 0:

                self.soc[i + 1] = min(soc_max, self.soc[i] + (self.charged_energy_to_BESS[i] - self.from_BESS_to_load[i]) / size)

            # IF BESS IS DISCHARGING
            else:

                self.soc[i + 1] = max(soc_min, self.soc[i] + (self.discharged_energy_from_BESS[i] - self.from_BESS_to_load[i]) / size)

            total_energy = self.charged_energy_from_BESS[i] + np.abs(self.discharged_energy_from_BESS[i])
            actual_capacity = size * degradation(n_cycles_prev) / 100
            n_cycles = n_cycles_prev + total_energy / actual_capacity

        # EVALUATE THE NUMBER OF CYCLES DONE BY BESS
        total_charged = np.sum(self.charged_energy_from_BESS)
        total_discharged = np.sum(-self.discharged_energy_from_BESS)
        total_energy = total_charged + total_discharged

        from argparser_s import n_cycles

        n_cycles_prev = n_cycles
        actual_capacity = size * degradation(n_cycles_prev)/100

        n_cycles = total_energy / actual_capacity

        # EVALUATE THE REVENUES OBTAINED FOR EACH TIMESTEP t
        revenue_column = np.array(np.abs(self.discharged_energy_from_BESS) * self.PUN_timeseries   / 1000 -
                                      np.abs(self.charged_energy_from_grid_to_BESS) * self.PUN_timeseries * 1.1 / 1000
                                      + np.abs(self.discharged_from_pv) * self.PUN_timeseries / 1000
                                      + np.abs(self.from_pv_to_load) * self.PUN_timeseries * 1.1 / 1000
                                      + np.abs(self.from_BESS_to_load) * self.PUN_timeseries * 1.1  / 1000  )

        # EVALUATE REVENUES CONSIDERING TYPICAL DAYS FOR EACH MONTH
        num_settimane = 12
        ore_per_settimana = 24
        revenues_settimanali = np.zeros(num_settimane)

        for i in range(num_settimane):
            inizio = i * ore_per_settimana
            fine = inizio + ore_per_settimana
            revenues_settimanali[i] = np.sum(revenue_column[inizio:fine]) * 30

        revenues_finali = revenues_settimanali
        somma_revenues_finali = np.sum(revenues_finali)

        # EVALUATE THE REVENUES OBTAINED DURING THE OPTIMIZATION TIME WINDOW
        total_revenue = somma_revenues_finali

        # CORRECT THE VALUES OF THE REVENUES IN ORDER TO MINIMIZE THE OBJECTIVE FUNCTION
        final_revenues = -total_revenue

        # DEFINE THE OUTPUT OF THE OPTIMIZATION PROBLEM
        out["F"] = [final_revenues]



