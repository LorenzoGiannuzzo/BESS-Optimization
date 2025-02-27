""" BESS Optimization using NSGA-III Algorithm

    __author__ = "Lorenzo Giannuzzo"
    __maintainer__ = "Lorenzo Giannuzzo"
    __email__ = "lorenzo.giannuzzo@polito.it"
    __status__ = "in progress"
    __version__ = "v0.2.1"
    __license__ = "MIT"

Last Update of current code: 27/02/2025 """

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
        self.PUN_timeseries_sell = Economic_parameters_s.PUN_timeseries_sell[:, 1]
        self.PUN_timeseries_buy = Economic_parameters_s.PUN_timeseries_buy[:, 1]
        self.c_func, self.d_func = charge_rate_interpolated_func, discharge_rate_interpolated_func

        # DEFINE OBJECTIVE FUNCTION PARAMETERS - SoC, Charged/Discharged Energy (from BESS)
        self.soc = np.zeros((len(self.PUN_timeseries_sell)))
        self.charged_energy_to_BESS = np.zeros((len(self.PUN_timeseries_sell)))
        self.discharged_energy_from_BESS = np.zeros((len(self.PUN_timeseries_sell)))
        self.load_self_consumption = np.zeros((len(self.PUN_timeseries_sell)))
        self.from_pv_to_load = np.zeros((len(self.PUN_timeseries_sell)))
        self.from_BESS_to_load = np.zeros((len(self.PUN_timeseries_sell)))
        self.load_decision = np.zeros((len(self.PUN_timeseries_sell)))

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
        #bess_model = BESS_model(self.time_window, self.PUN_timeseries_sell, self.soc, self.size, self.c_func,
                                    #self.d_func)

        # GET CHARGED/DISCHARGED VALUES FROM BESS MODEL RUN
        #self.charged_energy_from_BESS, self.discharged_energy_from_BESS = bess_model.run_simulation(self.c_d_timeseries)

        self.charged_energy_from_BESS = np.zeros(len(self.c_d_timeseries))
        self.discharged_energy_from_BESS = np.zeros(len(self.c_d_timeseries))

        for i in range(len(self.c_d_timeseries)):

            if self.c_d_timeseries[i] > 0:

                self.charged_energy_from_BESS[i] = self.c_d_timeseries[i] * size

                self.discharged_energy_from_BESS[i] = 0

            elif self.c_d_timeseries[i] < 0:

                self.discharged_energy_from_BESS[i] = self.c_d_timeseries[i] * size

                self.charged_energy_from_BESS[i] = 0

            else:

                self.charged_energy_from_BESS[i] = 0

                self.discharged_energy_from_BESS[i] = 0

        # INITIALIZE VARIABLES
        total_available_energy = np.zeros((len(self.PUN_timeseries_sell)))
        self.taken_from_pv = np.zeros((len(self.PUN_timeseries_sell)))
        self.charged_energy_from_grid_to_BESS = np.zeros((len(self.PUN_timeseries_sell)))
        self.discharged_from_pv = np.zeros((len(self.PUN_timeseries_sell)))

        # EXECUTE THE UPDATE FOR EACH i-th TIMESTEP OF ALL THE ENERGY VECTORS. EVALUATING ENERGY BALANCES
        from argparser_s import n_cycles
        from argparser_s import soc_max, soc_min

        for i in range(self.time_window - 1):

            # UPDATE SOC MAX BASED ON ITS ACTUAL AND PAST DEGRADATION
            n_cycles_prev = n_cycles
            max_capacity = degradation(n_cycles_prev) / 100
            soc_max = min(soc_max, max_capacity)

            # EVALUATE THE TOTAL AVAILABLE ENERGY TO PERFORM SELF-CONSUMPTION AT THE i-th TIMESTEP
            # WHICH IS EVALUATED AS THE BES SOC ABOVE THE LOWER LIMIT VALUE * BESS_SIZE * P/E_RATIO [kWh] +
            # THE ENERGY PRODUCED FROM PV (Considering also WHAT THE BESS CAN DO, which is BESS_SIZE*P/E_RATIO
            # In other words is the energy that the BESS can give to the load + the energy produced by PV tp perform
            # self-consumption

            total_available_energy[i] = np.minimum((self.soc[i] - soc_min) * size, size * self.d_func(self.soc[i])) + self.production[i]

            assert total_available_energy[i] >= 0, "Total Available Energy is negative (1).\n\n"

            from argparser_s import self_consumption

            # EVALUATE SELF CONSUMPTION
            if self_consumption == 'True':

                self.load_self_consumption[i] = np.minimum(self.load[i], total_available_energy[i])

                assert self.load_self_consumption[i] >= 0, "Total self consumption is negative (1).\n\n"

            else:

                self.load_self_consumption[i] = self.load_decision[i] * np.minimum(self.load[i], total_available_energy[i])

                assert self.load_self_consumption[i] >= 0, "Total self consumption is negative (2).\n\n"

            # EVALUATE THE ENERGY THAT GOES FROM PV TO LOAD FOR SELF-CONSUMPTION PURPOSES
            self.from_pv_to_load[i] = np.minimum(self.load_self_consumption[i], self.production[i])

            assert self.from_pv_to_load[i] >= 0, "Energy from PV to the load is negative (1).\n\n"

            # EVALUATE THE ENERGY TAKEN BY THE BESS FROM PV
            self.taken_from_pv[i] = np.minimum(self.production[i] - self.from_pv_to_load[i], self.charged_energy_from_BESS[i])

            assert self.taken_from_pv[i] >= 0, "Energy taken from PV to the BESS is negative (1).\n\n"

            # APPLY BESS CONSTRAINTS ON ENERGY TAKEN FROM PV

            # HOW MUCH ENERGY BESS CAN TAKE IN THE TIME-STAMP
            self.taken_from_pv[i] = np.minimum(self.taken_from_pv[i], size * self.c_func(self.soc[i]))

            assert self.taken_from_pv[i] >= 0, "Energy taken from PV to the BESS is negative (2).\n\n"

            # HOW MUCH ENERGY THE BESS CAN CHARGE BASED ON THE CAP OF SOC_MAX
            self.taken_from_pv[i] = np.minimum(self.taken_from_pv[i], np.maximum((soc_max - self.soc[i]) * size, 0.0))

            assert self.taken_from_pv[i] >= 0, "Energy taken from PV to the BESS is negative (3).\n\n"

            # EVALUATE THE ENERGY USED TO CHARGE THE BESS TAKEN FROM THE GRID (IF CHARGED_ENERGY_FROM_BESS IS NEGATIVE,
            # MEANING THAT THE BESS IS DISCHARGING, THIS VALUE IS = 0
            self.charged_energy_from_grid_to_BESS[i] = np.maximum(self.charged_energy_from_BESS[i] -
                                                                  self.taken_from_pv[i], 0.0)

            assert self.charged_energy_from_grid_to_BESS[i] >= 0, "Energy taken from Grid to BESS is negative (1).\n\n"

            # EVALUATE THE ENERGY THAT GOES FROM THE BESS TO THE LOAD FOR SELF-CONSUMPTION
            self.from_BESS_to_load[i] = np.maximum(self.load_self_consumption[i] - self.from_pv_to_load[i], 0.0)

            assert self.from_BESS_to_load[i] >= 0, "Energy from BESS to load is negative (1).\n\n"

            # APPLY BESS CONSTRAINTS ON ENERGY SENT TO THE LOAD

            # HOW MUCH ENERGY THE BESS CAN GIVE
            self.from_BESS_to_load[i] = np.minimum(self.from_BESS_to_load[i], size * self.d_func(self.soc[i]))

            assert self.from_BESS_to_load[i] >= 0, "Energy from BESS to load is negative (2).\n\n"

            # HOW MUCH ENERGY THE BESS CAN GIVE BASED ON THE CAP OF SOC_MIN
            self.from_BESS_to_load[i] = np.minimum(self.from_BESS_to_load[i], (self.soc[i] - soc_min) * size)

            assert self.from_BESS_to_load[i] >= 0, "Energy from BESS to load is negative (3).\n\n"

            # UPDATE THE ENERGY THAT THE BESS WANT TO CHARGE AS SUM OF THE ONE CHARGED FROM GRID TO BESS AND THE ENERGY
            # TAKEN FROM PV TO THE BESS
            self.charged_energy_to_BESS[i] = self.charged_energy_from_grid_to_BESS[i] + self.taken_from_pv[i]

            assert self.charged_energy_from_BESS[i] >= 0, "Energy that goes inside the BESS is negative (1) .\n\n"

            # UPDATE THE ENERGY DISCHARGED FROM PV DIRECTLY TO THE GRID REDUCING ITS ORIGINAL VALUE BY THE ONE THAT
            # GOES FROM PV TO THE BESS AND FROM THE PV TO THE LOAD
            self.discharged_from_pv[i] = np.minimum(-self.production[i] + self.taken_from_pv[i] +
                                                    self.from_pv_to_load[i], 0.0) # NEGATIVE VALUE

            assert self.discharged_from_pv[i] <= 0, "Energy discharged from PV is positive (1).\n\n"

            # APPLY BESS CONSTRAINTS ON ENERGY DISCHARGED FROM BESS

            # HOW MUCH ENERGY BESS CAN DISCHARGE IN THE TIME-STAMP
            self.discharged_energy_from_BESS[i] = -np.minimum(np.abs(self.discharged_energy_from_BESS[i]),size * self.d_func(self.soc[i]))

            assert self.discharged_energy_from_BESS[i] <= 0, "Energy discharged from BESS is positive (1).\n\n"

            # HOW MUCH ENERGY THE BESS DISCHARGE GIVE BASED ON THE CAP OF SOC_MIN
            self.discharged_energy_from_BESS[i] = np.maximum(self.discharged_energy_from_BESS[i], (soc_min - self.soc[i])*size)

            assert self.discharged_energy_from_BESS[i] <= 0, "Energy discharged from BESS is positive (2).\n\n"

            # APPLY POD CONSTRAINTS TO ENERGY VECTORS

            # IF POD POWER IS EXCEEDED WHILE TAKING ENERGY FROM THE GRID
            if self.charged_energy_from_grid_to_BESS[i] + self.load[i] > POD_power:

                # FIRST OF ALL, LIMIT THE LOAD IF EVEN THE LOAD EXCEED THE POD POWER LIMITS (NOT CONTROLLABLE)
                self.load[i] = np.minimum(POD_power, self.load[i])

                assert self.load[i] >= 0, "Load is negative (1).\n\n"

                # THEN LIMIT ALSO THE ENERGY CHARGED FROM GRID TO BESS (CONTROLLABLE)
                self.charged_energy_from_grid_to_BESS[i] = np.maximum(POD_power - self.load[i], 0.0)

                assert self.charged_energy_from_grid_to_BESS[i] >= 0, "Charged Energy from grid to BESS is negative (2).\n\n"

            # IF POD POWER IS EXCEEDED WHILE DISCHARGING ENERGY TO THE GRID
            if -np.abs(self.discharged_from_pv[i]) - np.abs(self.discharged_energy_from_BESS[i]) < -POD_power:

                # FIRST OF ALL LIMIT TGE ENERGY DISCHARGED FROM PV IF IT EXCEED ALONE THE POD POWER LIMIT (NOT
                # CONTROLLABLE)
                self.discharged_from_pv[i] = np.maximum(-POD_power, -np.abs(self.discharged_from_pv[i]))

                assert self.discharged_from_pv[i] <= 0, "Energy discharged from PV is positive (2).\n\n"

                # THEN ALSO LIMIT THE ENERGY DISCHARGED FROM BESS TO THE GRID (CONTROLLABLE)
                self.discharged_energy_from_BESS[i] = -np.maximum(POD_power - np.abs(self.discharged_from_pv[i]), 0.0)

                assert self.discharged_energy_from_BESS[i] <= 0, "Energy discharged from BESS is positive (3).\n\n"

            # AFTER APPLYING POD CONSTRAINTS, LOAD, CHARGED ENERGY FROM BESS, DISCHARGED FROM PV AND DISCHARGED
            # FROM BESS COULD BE CHANGED

            total_available_energy[i] = np.minimum((self.soc[i] - soc_min) * size, size * self.d_func(self.soc[i])) + \
                                        self.production[i]

            assert total_available_energy[i] >= 0, "Total Available Energy is negative (2).\n\n"

            from argparser_s import self_consumption

            # EVALUATE SELF CONSUMPTION
            if self_consumption == 'True':

                self.load_self_consumption[i] = np.minimum(self.load[i], total_available_energy[i])

                assert self.load_self_consumption[i] >= 0, "Total self consumption is negative (3).\n\n"

            else:

                self.load_self_consumption[i] = self.load_decision[i] * np.minimum(self.load[i],
                                                                                   total_available_energy[i])

                assert self.load_self_consumption[i] >= 0, "Total self consumption is negative (4).\n\n"

            # EVALUATE THE ENERGY THAT GOES FROM PV TO LOAD FOR SELF-CONSUMPTION PURPOSES
            self.from_pv_to_load[i] = np.minimum(self.load_self_consumption[i], self.production[i])

            assert self.from_pv_to_load[i] >= 0, "Energy from PV to the load is negative (2).\n\n"

            # EVALUATE THE ENERGY TAKEN BY THE BESS FROM PV
            self.taken_from_pv[i] = np.minimum(self.production[i] - self.from_pv_to_load[i],
                                               self.charged_energy_from_BESS[i])

            assert self.taken_from_pv[i] >= 0, "Energy taken from PV to the BESS is negative (4).\n\n"

            # APPLY BESS CONSTRAINTS ON ENERGY TAKEN FROM PV

            # HOW MUCH ENERGY BESS CAN TAKE IN THE TIME-STAMP
            self.taken_from_pv[i] = np.minimum(self.taken_from_pv[i], size * self.c_func(self.soc[i]))

            assert self.taken_from_pv[i] >= 0, "Energy taken from PV to the BESS is negative (5).\n\n"

            # HOW MUCH ENERGY THE BESS CAN CHARGE BASED ON THE CAP OF SOC_MAX
            self.taken_from_pv[i] = np.minimum(self.taken_from_pv[i], np.maximum((soc_max - self.soc[i]) * size, 0.0))

            assert self.taken_from_pv[i] >= 0, "Energy taken from PV to the BESS is negative (6).\n\n"

            # EVALUATE THE ENERGY USED TO CHARGE THE BESS TAKEN FROM THE GRID (IF CHARGED_ENERGY_FROM_BESS IS NEGATIVE,
            # MEANING THAT THE BESS IS DISCHARGING, THIS VALUE IS = 0
            self.charged_energy_from_grid_to_BESS[i] = np.maximum(self.charged_energy_from_BESS[i] -
                                                                  self.taken_from_pv[i], 0.0)

            assert self.charged_energy_from_grid_to_BESS[i] >= 0, "Energy taken from Grid to BESS is negative (3).\n\n"

            # EVALUATE THE ENERGY THAT GOES FROM THE BESS TO THE LOAD FOR SELF-CONSUMPTION
            self.from_BESS_to_load[i] = np.maximum(self.load_self_consumption[i] - self.from_pv_to_load[i], 0.0)

            assert self.from_BESS_to_load[i] >= 0, "Energy from BESS to load is negative (4).\n\n"

            # APPLY BESS CONSTRAINTS ON ENERGY SENT TO THE LOAD

            # HOW MUCH ENERGY THE BESS CAN GIVE
            self.from_BESS_to_load[i] = np.minimum(self.from_BESS_to_load[i], size * self.d_func(self.soc[i]))

            assert self.from_BESS_to_load[i] >= 0, "Energy from BESS to load is negative (5).\n\n"

            # HOW MUCH ENERGY THE BESS CAN GIVE BASED ON THE CAP OF SOC_MIN
            self.from_BESS_to_load[i] = np.minimum(self.from_BESS_to_load[i], (self.soc[i] - soc_min) * size)

            assert self.from_BESS_to_load[i] >= 0, "Energy from BESS to load is negative (6).\n\n"

            # UPDATE THE ENERGY THAT THE BESS WANT TO CHARGE AS SUM OF THE ONE CHARGED FROM GRID TO BESS AND THE ENERGY
            # TAKEN FROM PV TO THE BESS
            self.charged_energy_to_BESS[i] = self.charged_energy_from_grid_to_BESS[i] + self.taken_from_pv[i]

            assert self.charged_energy_from_BESS[i] >= 0, "Energy that goes inside the BESS is negative (2).\n\n"

            # UPDATE THE ENERGY DISCHARGED FROM PV DIRECTLY TO THE GRID REDUCING ITS ORIGINAL VALUE BY THE ONE THAT
            # GOES FROM PV TO THE BESS AND FROM THE PV TO THE LOAD
            self.discharged_from_pv[i] = np.minimum(-self.production[i] + self.taken_from_pv[i] +
                                                    self.from_pv_to_load[i], 0.0)  # NEGATIVE VALUE

            assert self.discharged_from_pv[i] <= 0, "Energy discharged from PV is positive (2).\n\n"

            # APPLY BESS CONSTRAINTS ON ENERGY DISCHARGED FROM BESS

            # HOW MUCH ENERGY BESS CAN DISCHARGE IN THE TIME-STAMP
            self.discharged_energy_from_BESS[i] = -np.minimum(np.abs(self.discharged_energy_from_BESS[i]),
                                                              size * self.d_func(self.soc[i]))

            assert self.discharged_energy_from_BESS[i] <= 0, "Energy discharged from BESS is positive (3).\n\n"

            # HOW MUCH ENERGY THE BESS DISCHARGE GIVE BASED ON THE CAP OF SOC_MIN
            self.discharged_energy_from_BESS[i] = np.maximum(self.discharged_energy_from_BESS[i],
                                                             (soc_min - self.soc[i]) * size)

            assert self.discharged_energy_from_BESS[i] <= 0, "Energy discharged from BESS is positive (4).\n\n"

            # UPDATE SOC

            # IF BESS IS CHARGING
            if self.c_d_timeseries[i] > 0:

                self.soc[i + 1] = min(soc_max, self.soc[i] + (self.charged_energy_to_BESS[i] -
                                                              np.abs(self.from_BESS_to_load[i])) / size)
                self.discharged_energy_from_BESS[i] = 0

                # assert self.soc[i+1] >= self.soc[i], "SoC is decreasing instead of incresing.\n\n"

            # IF BESS IS DISCHARGING
            elif self.c_d_timeseries[i] < 0:

                self.soc[i + 1] = max(soc_min, self.soc[i] - (np.abs(self.discharged_energy_from_BESS[i]) +
                                                              np.abs(self.from_BESS_to_load[i])) / size)

                # assert self.soc[i+1] <= self.soc[i], "SoC is increasing instead of decreasing.\n\n"

                self.charged_energy_from_grid_to_BESS[i] = 0
            else:

                self.discharged_energy_from_BESS[i] = 0
                self.charged_energy_from_BESS[i] = 0
                self.soc[i+1] = self.soc[i]

            total_energy = self.charged_energy_from_grid_to_BESS[i] + np.abs(self.discharged_energy_from_BESS[i]) + np.abs(self.from_BESS_to_load[i])
            actual_capacity = size * degradation(n_cycles_prev) / 100
            n_cycles = n_cycles_prev + total_energy / actual_capacity

        # EVALUATE THE NUMBER OF CYCLES DONE BY BESS
        total_charged = np.sum(self.charged_energy_from_BESS)
        total_discharged = np.sum(-self.discharged_energy_from_BESS)
        additional = np.sum(np.abs(self.from_BESS_to_load))
        total_energy = total_charged + total_discharged + additional

        from argparser_s import n_cycles

        n_cycles_prev = n_cycles
        actual_capacity = size * degradation(n_cycles_prev)/100

        n_cycles = total_energy / actual_capacity

        # EVALUATE THE REVENUES OBTAINED FOR EACH TIMESTEP t
        revenue_column = (np.array(np.abs(self.discharged_energy_from_BESS) * self.PUN_timeseries_sell / 1000
                                   - np.abs(self.charged_energy_from_grid_to_BESS) * self.PUN_timeseries_buy * 1.2 / 1000
                                      + np.abs(self.discharged_from_pv) * self.PUN_timeseries_sell / 1000
                                      + np.abs(self.from_pv_to_load) * self.PUN_timeseries_buy * 1.2 / 1000
                                      + np.abs(self.from_BESS_to_load) * self.PUN_timeseries_buy * 1.2/ 1000)
                          - ( np.abs(self.load) - np.abs(self.from_pv_to_load) - np.abs(self.from_BESS_to_load) ) * self.PUN_timeseries_sell * 1.2 / 1000)

        # EVALUATES TYPICAL DAYS REVENUES
        num_settimane = 12  # Number of weeks to evaluate
        ore_per_settimana = 24  # Hours per week
        revenues_settimanali = np.zeros(num_settimane)  # Initialize weekly revenues array

        # EVALUATE REVENUES
        for i in range(num_settimane):
            inizio = i * ore_per_settimana  # Start index for the week
            fine = inizio + ore_per_settimana  # End index for the week
            revenues_settimanali[i] = np.sum(revenue_column[inizio:fine]) * 30  # Calculate weekly revenues

        revenues_finali = revenues_settimanali

        # EVALUATE TOTAL REVENUES
        revv = np.sum(revenues_finali)

        # EVALUATE THE REVENUES OBTAINED DURING THE OPTIMIZATION TIME WINDOW
        total_revenue = np.sum(revenue_column)

        # CORRECT THE VALUES OF THE REVENUES IN ORDER TO MINIMIZE THE OBJECTIVE FUNCTION
        final_revenues = -revv

        print(final_revenues)

        # DEFINE THE OUTPUT OF THE OPTIMIZATION PROBLEM
        out["F"] = [final_revenues]



