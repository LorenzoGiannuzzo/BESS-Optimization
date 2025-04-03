""" BESS Optimization using NSGA-III Algorithm

    __author__ = "Lorenzo Giannuzzo"
    __maintainer__ = "Lorenzo Giannuzzo"
    __email__ = "lorenzo.giannuzzo@polito.it"
    __status__ = "in progress"
    __version__ = "v0.2.1"
    __license__ = "MIT"

Last Update of current code: 02/04/2025 """

# IMPORTING LIBRARIES AND MODULES FROM PROJECT
import numpy as np  # Numerical operations
import json  # JSON file handling
from pymoo.core.problem import StarmapParallelization  # Parallelization for optimization
from multiprocessing import Pool, cpu_count  # Multiprocessing utilities
from objective_function_l import Revenues  # Objective function for revenue calculation
from configuration_l import pop_size, soc_0, time_window, plot  # Configuration parameters
from BESS_model_l import charge_rate_interpolated_func, discharge_rate_interpolated_func, size, technology, BESS_model  # BESS model functions and parameters
from Economic_parameters_l import PUN_timeseries_sell, PUN_timeseries_buy  # Economic parameters for pricing
from Optimizer_l import Optimizer  # Optimization class
from argparser_l import output_json_path, range_str, soc_min, power_energy, POD_power  # Argument parser parameters
from Plots_l import EnergyPlots  # Plotting utilities
from PV_l import pv_production  # Photovoltaic production data
import subprocess
import logging
from logger import setup_logger

setup_logger()

# CREATION OF CLASS MAIN
class Main:

    # CLASS ATTRIBUTES DEFINITION
    def __init__(self, multiprocessing=True):
        """
        Initializes the main object. Creates an instance of the objective function (Revenues) and the optimizer (Optimizer).
        Args:
            multiprocessing (bool): Flag to enable or disable multiprocessing.
        """

        # DEFINE MULTIPROCESSING (TRUE OR FALSE)
        self.multiprocessing = multiprocessing

        # INITIALIZE MULTIPROCESSING
        if self.multiprocessing:
            n_processes = cpu_count() - 1  # Number of processes to use
            self.pool = Pool(processes=n_processes)  # Create a pool of worker processes
            runner = StarmapParallelization(self.pool.starmap)  # Set up parallelization runner
            self.objective_function = Revenues(elementwise_runner=runner, elementwise=True)  # Initialize objective function for revenues
        else:
            self.objective_function = Revenues()  # Initialize without parallelization

        # SET OPTIMIZER
        self.optimizer = Optimizer(objective_function=self.objective_function, pop_size=pop_size,
                                   multiprocessing=multiprocessing)  # Initialize optimizer with the objective function

    def run_optimization(self):
        """
        Executes the optimization, applies physical constraints, calculates revenues, and generates plots.
        """
        # GET SOLUTION FROM OPTIMIZATION TASK
        solution = self.optimizer.maximize_revenues()  # Run optimization to maximize revenues

        # CLOSE POOL IF MULTIPROCESSING IS TRUE
        if self.multiprocessing:
            self.pool.close()  # Close the pool of processes

        # SAVE OPTIMIZATION HISTORY
        if plot:
            self.history = solution.history  # Store optimization history for plotting

        # GET CHARGED/DISCHARGED ENERGY FROM SOLUTION
        c_d_timeseries = solution.X[:time_window]

        # Extract charge/discharge time series from solution
        load_decision = solution.X[time_window:2*time_window]

        # APPLY PHYSICAL CONSTRAINTS
        (soc, charged_energy, discharged_energy, c_d_timeseries, taken_from_grid, discharged_from_pv,
         taken_from_pv, n_cycler,load_self_consumption,from_pv_to_load, from_BESS_to_load) =\
            self.apply_physical_constraints(c_d_timeseries, load_decision)  # Apply constraints to the solution

        # DEFINE CLASS ATTRIBUTES AS CONSTRAINED SOLUTION OBTAINED FORM OPTIMIZATION
        self.c_d_timeseries_final = c_d_timeseries  # Final charge/discharge time series
        self.soc = soc  # State of charge
        self.charged_energy = charged_energy  # Charged energy time series
        self.discharged_energy = discharged_energy  # Discharged energy time series
        self.taken_from_grid = taken_from_grid  # Energy taken from the grid
        self.taken_from_pv = taken_from_pv  # Energy taken from PV
        self.discharged_from_pv = discharged_from_pv  # Energy discharged from PV
        self.n_cycler = n_cycler  # Number of cycles
        self.load_self_consumption = load_self_consumption
        self.from_pv_to_load = from_pv_to_load
        self.from_BESS_to_load = from_BESS_to_load



        # GET LOAD DATA
        from Load_l import data

        # CALCULATE AND PRINT REVENUES
        self.calculate_and_print_revenues(self.charged_energy, self.discharged_energy, self.taken_from_grid,
                                          self.discharged_from_pv, self.from_pv_to_load, self.from_BESS_to_load, data)

        self.load = data
        self.pv_production = pv_production['P']

        # Calculate and display revenues

        # GENERATE PLOTS IF PLOT FLAG IS TRUE
        if plot:
            self.plot_results(soc, charged_energy, discharged_energy, c_d_timeseries, PUN_timeseries_sell[:, 1],
                              taken_from_grid, taken_from_pv, discharged_from_pv, load_self_consumption,from_pv_to_load,
                              from_BESS_to_load, data)  # Generate plots for the results

    # CONSTRAINTS FUNCTION DEFINITION
    @staticmethod
    def apply_physical_constraints(c_d_timeseries, load_decision):

        from Load_l import data
        from argparser_l import n_cycles  # Import number of cycles
        from BESS_model_l import degradation
        from argparser_l import n_cycles
        from argparser_l import soc_max, soc_min

        # GET CHARGE/DISCHARGE INTERPOLATED FUNCTIONS
        c_func = charge_rate_interpolated_func
        d_func = discharge_rate_interpolated_func

        # INITIALIZE FIRST PARAMETERS
        soc = [soc_0] * time_window
        soc[0] = soc_0
        #
        # bess_model = BESS_model(time_window, PUN_timeseries_sell, soc, size, c_func, d_func)  # Create BESS model instance
        #
        # # Run simulation to get energy values
        # charged_energy_from_BESS, discharged_energy_from_BESS = bess_model.run_simulation(c_d_timeseries)

        # GET VARIABLES
        production = pv_production['P']
        load = data

        # INITIALIZE VARIABLES
        n_cycler = [0] * time_window
        n_cycler[0] = n_cycles
        total_available_energy = [0.0] * time_window
        self_consumption = [0.0] * time_window
        from_pv_to_load = [0.0] * time_window
        from_BESS_to_load = [0.0] * time_window
        load_self_consumption = [0.0] * time_window
        taken_from_pv = [0.0] * time_window
        charged_energy_from_grid_to_BESS = [0.0] * time_window
        discharged_from_pv = [0.0] * time_window
        remaining_pv = [0.0] * time_window
        remaining_load = [0.0] * time_window
        charged_energy_from_BESS = [0.0] * time_window
        discharged_energy_from_BESS = [0.0] * time_window

        # EXECUTE THE UPDATE FOR EACH i-th TIMESTEP OF ALL THE ENERGY VECTORS. EVALUATING ENERGY BALANCES
        from argparser_l import n_cycles
        from argparser_l import soc_max, soc_min

        for i in range(time_window - 1):

            # UPDATE SOC MAX BASED ON ITS ACTUAL AND PAST DEGRADATION
            n_cycles_prev = n_cycles
            max_capacity = degradation(n_cycles_prev) / 100
            soc_max = min(soc_max, max_capacity)

            # EVALUATE THE TOTAL AVAILABLE ENERGY TO PERFORM SELF-CONSUMPTION AT THE i-th TIMESTEP
            # WHICH IS EVALUATED AS THE BES SOC ABOVE THE LOWER LIMIT VALUE * BESS_SIZE * P/E_RATIO [kWh] +
            # THE ENERGY PRODUCED FROM PV (Considering also WHAT THE BESS CAN DO, which is BESS_SIZE*P/E_RATIO
            # In other words is the energy that the BESS can give to the load + the energy produced by PV tp perform
            # self-consumption

            if c_d_timeseries[i] > 0:

                charged_energy_from_BESS[i] = np.minimum(c_d_timeseries[i] * size,
                                                              c_func(soc[i]) * size)
                charged_energy_from_BESS[i] = np.minimum(charged_energy_from_BESS[i],
                                                              np.maximum((soc_max - soc[i]) * size, 0.0))

                assert charged_energy_from_BESS[
                           i] >= 0, f"Charged energy into BESS is negative. {c_d_timeseries[i]}\n\n {c_func(soc[i])}\n\n {soc_max - soc[i]}"

                discharged_energy_from_BESS[i] = 0

            elif c_d_timeseries[i] < 0:

                discharged_energy_from_BESS[i] = np.maximum(c_d_timeseries[i] * size,
                                                                 -d_func(soc[i]) * size)
                discharged_energy_from_BESS[i] = np.maximum(discharged_energy_from_BESS[i],
                                                                 np.minimum((soc_min - soc[i]) * size, 0.0))

                charged_energy_from_BESS[i] = 0

            else:

                charged_energy_from_BESS[i] = 0
                discharged_energy_from_BESS[i] = 0

            # LOAD ESTIMATION ------------------------------------------------------------------------------------------

            total_available_energy[i] = np.minimum(np.maximum((soc[i] - soc_min), 0.0) * size,
                                                   size * d_func(soc[i])) + production[i]

            assert total_available_energy[i] >= 0, "Total Available Energy is negative (1).\n\n"

            from argparser_l import self_consumption

            # (A) EVALUATE SELF CONSUMPTION
            if self_consumption == 'True':

                load_self_consumption[i] = np.minimum(load[i], total_available_energy[i])

                assert load_self_consumption[i] >= 0, "Total self consumption is negative (A).\n\n"

            else:

                load_self_consumption[i] = load_decision[i] * np.minimum(load[i],
                                                                                   total_available_energy[i])

                assert load_self_consumption[i] >= 0, "Total self consumption is negative (A-2).\n\n"

            # (B) EVALUATE THE ENERGY THAT GOES FROM PV TO LOAD FOR SELF-CONSUMPTION PURPOSES
            from_pv_to_load[i] = np.minimum(load_self_consumption[i], production[i])

            assert from_pv_to_load[i] >= 0, "Energy from PV to the load is negative (B).\n\n"

            # (C) EVALUATE THE ENERGY THAT'S LEFT TO THE PV
            remaining_pv[i] = production[i] - from_pv_to_load[i]

            assert remaining_pv[i] >= 0, "Energy remaining to PV is negative (C).\n\n"

            # (D) EVALUATE THE ENERGY THAT'S LEFT ON LOAD
            remaining_load[i] = load[i] - load_self_consumption[i]

            assert remaining_load[i] >= 0, "Energy remaining to load is negative (D). \n\n"

            # (E) EVALUATE THE ENERGY THAT GOES FROM THE BESS TO THE LOAD FOR SELF-CONSUMPTION
            from_BESS_to_load[i] = np.maximum(load_self_consumption[i] - from_pv_to_load[i], 0.0)

            assert from_BESS_to_load[i] >= 0, "Energy from BESS to load is negative (E).\n\n"

            # (F) APPLY BESS CONSTRAINTS ON ENERGY SENT TO THE LOAD

            # (F-1) HOW MUCH ENERGY THE BESS CAN GIVE
            from_BESS_to_load[i] = np.minimum(from_BESS_to_load[i], size * d_func(soc[i]))

            assert from_BESS_to_load[i] >= 0, "Energy from BESS to load is negative (F-1).\n\n"

            # (F-2) HOW MUCH ENERGY THE BESS CAN GIVE BASED ON THE CAP OF SOC_MIN
            from_BESS_to_load[i] = np.minimum(from_BESS_to_load[i], (soc[i] - soc_min) * size)

            assert from_BESS_to_load[
                       i] >= 0, f"Energy from BESS to load is negative (F-2).\n\n {soc[i]}\n\n {soc_min}"

            # ----------------------------------------------------------------------------------------------------------

            # BESS ESTIMATION ------------------------------------------------------------------------------------------

            # (G) EVALUATE THE ENERGY TAKEN BY THE BESS FROM PV
            taken_from_pv[i] = np.minimum(remaining_pv[i], charged_energy_from_BESS[i])

            assert taken_from_pv[i] >= 0, "Energy taken from PV to the BESS is negative (G).\n\n"

            # (H) APPLY BESS CONSTRAINTS ON ENERGY TAKEN FROM PV

            # (H-1) HOW MUCH ENERGY BESS CAN TAKE IN THE TIME-STAMP
            taken_from_pv[i] = np.minimum(taken_from_pv[i], size * c_func(soc[i]))

            assert taken_from_pv[i] >= 0, "Energy taken from PV to the BESS is negative (H-1).\n\n"

            # (H-2) HOW MUCH ENERGY THE BESS CAN CHARGE BASED ON THE CAP OF SOC_MAX
            taken_from_pv[i] = np.minimum(taken_from_pv[i],
                                               np.maximum((soc_max - soc[i]) * size + from_BESS_to_load[i],
                                                          0.0))

            assert taken_from_pv[i] >= 0, "Energy taken from PV to the BESS is negative (H-2).\n\n"

            # (I) EVALUATE THE ENERGY USED TO CHARGE THE BESS TAKEN FROM THE GRID (IF CHARGED_ENERGY_FROM_BESS IS NEGATIVE,
            # MEANING THAT THE BESS IS DISCHARGING, THIS VALUE IS = 0
            charged_energy_from_grid_to_BESS[i] = np.maximum(charged_energy_from_BESS[i] -
                                                                  taken_from_pv[i], 0.0)

            assert charged_energy_from_grid_to_BESS[i] >= 0, "Energy taken from Grid to BESS is negative (I).\n\n"

            # (EXTRA)

            if from_BESS_to_load[i] > 0:
                charged_energy_from_grid_to_BESS[i] = 0

            # (J) UPDATE THE ENERGY THAT THE BESS WANT TO CHARGE AS SUM OF THE ONE CHARGED FROM GRID TO BESS AND THE ENERGY
            # TAKEN FROM PV TO THE BESS
            charged_energy_from_BESS[i] = charged_energy_from_grid_to_BESS[i] + taken_from_pv[i]

            assert charged_energy_from_BESS[i] >= 0, "Energy that goes inside the BESS is negative (J) .\n\n"

            # (K) UPDATE THE ENERGY DISCHARGED FROM PV DIRECTLY TO THE GRID REDUCING ITS ORIGINAL VALUE BY THE ONE THAT
            # GOES FROM PV TO THE BESS AND FROM THE PV TO THE LOAD
            discharged_from_pv[i] = np.minimum(-remaining_pv[i] + taken_from_pv[i],
                                                    0.0)  # NEGATIVE VALUE

            assert discharged_from_pv[i] <= 0, "Energy discharged from PV is positive (K).\n\n"

            # (L) APPLY BESS CONSTRAINTS ON ENERGY DISCHARGED FROM BESS

            # (L-1) HOW MUCH ENERGY BESS CAN DISCHARGE IN THE TIME-STAMP
            discharged_energy_from_BESS[i] = -np.minimum(np.abs(discharged_energy_from_BESS[i]),
                                                              size * d_func(soc[i]))

            assert discharged_energy_from_BESS[i] <= 0, "Energy discharged from BESS is positive (L-1).\n\n"

            # (L-2) HOW MUCH ENERGY THE BESS DISCHARGE GIVE BASED ON THE CAP OF SOC_MIN
            discharged_energy_from_BESS[i] = np.maximum(discharged_energy_from_BESS[i],
                                                             (soc_min - soc[i]) * size + from_BESS_to_load[i])

            assert discharged_energy_from_BESS[i] <= 0, "Energy discharged from BESS is positive (L-2).\n\n"

            # (M) APPLY POD CONSTRAINTS TO ENERGY VECTORS

            # (M-1) IF POD POWER IS EXCEEDED WHILE TAKING ENERGY FROM THE GRID
            if charged_energy_from_grid_to_BESS[i] + load[i] > POD_power:
                # (M-1-1) FIRST OF ALL, LIMIT THE LOAD IF EVEN THE LOAD EXCEED THE POD POWER LIMITS (NOT CONTROLLABLE)
                load[i] = np.minimum(POD_power, load[i])

                assert load[i] >= 0, "Load is negative (M-1-1).\n\n"

                # (M-2-2) THEN LIMIT ALSO THE ENERGY CHARGED FROM GRID TO BESS (CONTROLLABLE)
                charged_energy_from_grid_to_BESS[i] = np.maximum(POD_power - load[i], 0.0)

                assert charged_energy_from_grid_to_BESS[
                           i] >= 0, "Charged Energy from grid to BESS is negative (M-2-2).\n\n"

            # (M-2) IF POD POWER IS EXCEEDED WHILE DISCHARGING ENERGY TO THE GRID
            if -np.abs(discharged_from_pv[i]) - np.abs(discharged_energy_from_BESS[i]) < -POD_power:
                # (M-2-1) FIRST OF ALL LIMIT TGE ENERGY DISCHARGED FROM PV IF IT EXCEED ALONE THE POD POWER LIMIT (NOT
                # CONTROLLABLE)
                discharged_from_pv[i] = np.maximum(-POD_power, -np.abs(discharged_from_pv[i]))

                assert discharged_from_pv[i] <= 0, "Energy discharged from PV is positive (M-2-1).\n\n"

                # (M-2-2) THEN ALSO LIMIT THE ENERGY DISCHARGED FROM BESS TO THE GRID (CONTROLLABLE)
                discharged_energy_from_BESS[i] = -np.maximum(POD_power - np.abs(discharged_from_pv[i]), 0.0)

                assert discharged_energy_from_BESS[i] <= 0, "Energy discharged from BESS is positive (M-2-2).\n\n"

            # AFTER APPLYING POD CONSTRAINTS, LOAD, CHARGED ENERGY FROM BESS, DISCHARGED FROM PV AND DISCHARGED
            # FROM BESS COULD BE CHANGED

            # LOAD ESTIMATION ------------------------------------------------------------------------------------------

            total_available_energy[i] = np.minimum(np.maximum((soc[i] - soc_min), 0.0) * size,
                                                   size * d_func(soc[i])) + production[i]

            assert total_available_energy[i] >= 0, "Total Available Energy is negative (1p).\n\n"

            from argparser_l import self_consumption

            # (N) EVALUATE SELF CONSUMPTION
            if self_consumption == 'True':

                load_self_consumption[i] = np.minimum(load[i], total_available_energy[i])

                assert load_self_consumption[i] >= 0, "Total self consumption is negative (N).\n\n"

            else:

                load_self_consumption[i] = load_decision[i] * np.minimum(load[i],
                                                                                   total_available_energy[i])

                assert load_self_consumption[i] >= 0, "Total self consumption is negative (N-2).\n\n"

            # (O) EVALUATE THE ENERGY THAT GOES FROM PV TO LOAD FOR SELF-CONSUMPTION PURPOSES
            from_pv_to_load[i] = np.minimum(load_self_consumption[i], production[i])

            assert from_pv_to_load[i] >= 0, "Energy from PV to the load is negative (O).\n\n"

            # (P) EVALUATE THE ENERGY THAT'S LEFT TO THE PV
            remaining_pv[i] = production[i] - from_pv_to_load[i]

            assert remaining_pv[i] >= 0, "Energy remaining to PV is negative (P).\n\n"

            # (Q) EVALUATE THE ENERGY THAT'S LEFT ON LOAD
            remaining_load[i] = load[i] - load_self_consumption[i]

            assert remaining_load[i] >= 0, "Energy remaining to load is negative (Q). \n\n"

            # (R) EVALUATE THE ENERGY THAT GOES FROM THE BESS TO THE LOAD FOR SELF-CONSUMPTION
            from_BESS_to_load[i] = np.maximum(load_self_consumption[i] - from_pv_to_load[i], 0.0)

            assert from_BESS_to_load[i] >= 0, "Energy from BESS to load is negative (R).\n\n"

            # (S) APPLY BESS CONSTRAINTS ON ENERGY SENT TO THE LOAD

            # (S-1) HOW MUCH ENERGY THE BESS CAN GIVE
            from_BESS_to_load[i] = np.minimum(from_BESS_to_load[i], size * d_func(soc[i]))

            assert from_BESS_to_load[i] >= 0, "Energy from BESS to load is negative (S-1).\n\n"

            # (S-2) HOW MUCH ENERGY THE BESS CAN GIVE BASED ON THE CAP OF SOC_MIN
            from_BESS_to_load[i] = np.minimum(from_BESS_to_load[i], (soc[i] - soc_min) * size)

            assert from_BESS_to_load[
                       i] >= 0, f"Energy from BESS to load is negative (S-2).\n\n {soc[i]}\n\n {soc_min}"

            # ----------------------------------------------------------------------------------------------------------

            # BESS ESTIMATION ------------------------------------------------------------------------------------------

            # (T) EVALUATE THE ENERGY TAKEN BY THE BESS FROM PV
            taken_from_pv[i] = np.minimum(remaining_pv[i], charged_energy_from_BESS[i])

            assert taken_from_pv[i] >= 0, "Energy taken from PV to the BESS is negative (T).\n\n"

            # (U) APPLY BESS CONSTRAINTS ON ENERGY TAKEN FROM PV

            # (U-1) HOW MUCH ENERGY BESS CAN TAKE IN THE TIME-STAMP
            taken_from_pv[i] = np.minimum(taken_from_pv[i], size * c_func(soc[i]))

            assert taken_from_pv[i] >= 0, "Energy taken from PV to the BESS is negative (U-1).\n\n"

            # (U-2) HOW MUCH ENERGY THE BESS CAN CHARGE BASED ON THE CAP OF SOC_MAX
            taken_from_pv[i] = np.minimum(taken_from_pv[i], np.maximum(
                (soc_max - soc[i]) * size + from_BESS_to_load[i], 0.0))

            assert taken_from_pv[i] >= 0, "Energy taken from PV to the BESS is negative (U-2).\n\n"

            # (V) EVALUATE THE ENERGY USED TO CHARGE THE BESS TAKEN FROM THE GRID (IF CHARGED_ENERGY_FROM_BESS IS NEGATIVE,
            # MEANING THAT THE BESS IS DISCHARGING, THIS VALUE IS = 0
            charged_energy_from_grid_to_BESS[i] = np.maximum(charged_energy_from_BESS[i] -
                                                                  taken_from_pv[i], 0.0)

            assert charged_energy_from_grid_to_BESS[
                       i] >= 0, "Energy taken from Grid to BESS is negative (V).\n\n"

            # (EXTRA)

            if from_BESS_to_load[i] > 0:
                charged_energy_from_grid_to_BESS[i] = 0

            # (Z) UPDATE THE ENERGY THAT THE BESS WANT TO CHARGE AS SUM OF THE ONE CHARGED FROM GRID TO BESS AND THE ENERGY
            # TAKEN FROM PV TO THE BESS
            charged_energy_from_BESS[i] = charged_energy_from_grid_to_BESS[i] + taken_from_pv[i]

            assert charged_energy_from_BESS[i] >= 0, "Energy that goes inside the BESS is negative (Z) .\n\n"

            # (W) UPDATE THE ENERGY DISCHARGED FROM PV DIRECTLY TO THE GRID REDUCING ITS ORIGINAL VALUE BY THE ONE THAT
            # GOES FROM PV TO THE BESS AND FROM THE PV TO THE LOAD
            discharged_from_pv[i] = np.minimum(-remaining_pv[i] + taken_from_pv[i],
                                                    0.0)  # NEGATIVE VALUE

            assert discharged_from_pv[i] <= 0, "Energy discharged from PV is positive (W).\n\n"

            # (X) APPLY BESS CONSTRAINTS ON ENERGY DISCHARGED FROM BESS

            # (X-1) HOW MUCH ENERGY BESS CAN DISCHARGE IN THE TIME-STAMP
            discharged_energy_from_BESS[i] = -np.minimum(np.abs(discharged_energy_from_BESS[i]),
                                                              size * d_func(soc[i]))

            assert discharged_energy_from_BESS[i] <= 0, "Energy discharged from BESS is positive (X-1).\n\n"

            # (X-2) HOW MUCH ENERGY THE BESS DISCHARGE GIVE BASED ON THE CAP OF SOC_MIN
            discharged_energy_from_BESS[i] = np.maximum(discharged_energy_from_BESS[i],
                                                             (soc_min - soc[i]) * size +
                                                             from_BESS_to_load[i])

            assert discharged_energy_from_BESS[i] <= 0, "Energy discharged from BESS is positive (X-2).\n\n"

            # (FINAL) UPDATE SOC

            # IF BESS IS CHARGING
            if c_d_timeseries[i] > 0:

                soc[i + 1] = min(soc_max, soc[i] + (charged_energy_from_BESS[i] -
                                                              np.abs(from_BESS_to_load[i])) / size)
                discharged_energy_from_BESS[i] = 0

                # assert self.soc[i+1] >= self.soc[i], "SoC is decreasing instead of incresing.\n\n"

            # IF BESS IS DISCHARGING
            elif c_d_timeseries[i] < 0:

                soc[i + 1] = max(soc_min, soc[i] - (np.abs(discharged_energy_from_BESS[i]) +
                                                              np.abs(from_BESS_to_load[i])) / size)

                # assert self.soc[i+1] <= self.soc[i], "SoC is increasing instead of decreasing.\n\n"

                charged_energy_from_grid_to_BESS[i] = 0
            else:

                discharged_energy_from_BESS[i] = 0
                charged_energy_from_BESS[i] = 0

                soc[i + 1] = soc[i] + (taken_from_pv[i] - np.abs(from_BESS_to_load[i])) / size

            total_energy = charged_energy_from_grid_to_BESS[i] + np.abs(
                discharged_energy_from_BESS[i]) + np.abs(from_BESS_to_load[i])
            actual_capacity = size * degradation(n_cycles_prev) / 100
            n_cycles = n_cycles_prev + total_energy / actual_capacity

        # EVALUATE THE NUMBER OF CYCLES DONE BY BESS
        total_charged = np.sum(charged_energy_from_BESS)
        total_discharged = np.sum(-np.array(discharged_energy_from_BESS))
        additional = np.sum(np.abs(from_BESS_to_load))
        total_energy = total_charged + total_discharged + additional

        from argparser_l import n_cycles

        n_cycles_prev = n_cycles
        actual_capacity = size * degradation(n_cycles_prev) / 100

        n_cycles = total_energy / actual_capacity

        return (soc, charged_energy_from_BESS, discharged_energy_from_BESS, c_d_timeseries, charged_energy_from_grid_to_BESS,
                discharged_from_pv, taken_from_pv, n_cycler, load_self_consumption, from_pv_to_load, from_BESS_to_load)

    def calculate_and_print_revenues(self, charged_energy, discharged_energy, taken_from_grid, discharged_from_pv,
                                     from_pv_to_load, from_BESS_to_load, load):

        from Economic_parameters_l import PUN_timeseries_sell, PUN_timeseries_buy

        revenue_column = (np.array(np.abs(discharged_energy) * PUN_timeseries_sell[:,1] / 1000
                                   - np.abs(taken_from_grid) * PUN_timeseries_buy[:,1] * 1.2 / 1000
                                      + np.abs(discharged_from_pv) * PUN_timeseries_sell[:,1] / 1000
                                      + np.abs(from_pv_to_load) * PUN_timeseries_buy[:,1] * 1.2 / 1000
                                      + np.abs(from_BESS_to_load) * PUN_timeseries_buy[:,1] * 1.2 / 1000)
                          - ( np.abs(load) - np.abs(from_pv_to_load) - np.abs(from_BESS_to_load) ) * PUN_timeseries_sell[:,1] * 1.2 / 1000)


        revenues_finali = revenue_column
        self.rev = revenue_column

        # EVALUATE TOTAL REVENUES
        revv = np.sum(revenues_finali)

        # DISPLAY TOTAL REVENUES
        print("\nRevenues for optimized time window [EUROs]:\n\n", revv)

    # DEFINE PLOT RESULTS FUNCTION
    def plot_results(self, soc, charged_energy, discharged_energy, c_d_energy, PUN_Timeseries, taken_from_grid,
                     taken_from_pv, discharged_from_pv, self_consumption, from_pv_to_load,from_BESS_to_load,load):

        from Load_l import data
        from BESS_model_l import charge_rate, discharge_rate

        # EXECUTE PLOTS
        if plot:
            plots = EnergyPlots(time_window, soc, charged_energy, discharged_energy, PUN_timeseries_sell[:, 1],
                                taken_from_grid, taken_from_pv, pv_production['P'], discharged_from_pv,
                                self_consumption, from_pv_to_load, from_BESS_to_load, np.array(data))
            plots.Total_View(num_values=time_window)
            plots.plot_daily_energy_flows(num_values=time_window)
            plots.plot_degradation()
            plots.c_d_plot(charge_rate, discharge_rate)


# MAIN EXECUTION
if __name__ == "__main__":

    main = Main(multiprocessing=True)  # Create Main instance with multiprocessing enabled
    main.run_optimization()

    # POSTPROCESSING
    X = []
    Y = []

    # IMPORT OPTIMIZATION HISTORY IF PLOT FLAG IS TRUE
    if plot:
        for j in range(len(main.history)):
            X.append([])  # Create a new list for each history entry
            for i in range(pop_size):
                X[j].append(main.history[j].pop[i].get('X'))  # Append X values from history

        for j in range(len(main.history)):
            Y.append([])  # Create a new list for each history entry
            for i in range(pop_size):
                Y[j].append(main.history[j].pop[i].get('f'))  # Append Y values from history

    X = np.array(X)
    Y = np.array(Y)

    # EXECUTE ADDITIONAL PLOTS IF PLOT FLAG IS TRUE
    if plot:
        EnergyPlots.total_convergence(len(main.history), time_window, pop_size, X, Y)

    # GET SOC VALUES
    SoC = main.soc

    # GET CHARGED/DISCHARGED ENERGY VALUES
    c_d_energy = main.c_d_timeseries_final

    # GET REVENUES VALUES
    revenues = main.rev
    data = []

    # OUTPUT CREATION AS .JSON FILE
    for i in range(len(PUN_timeseries_sell[:, 1])):
        entry = {
            # DATETIME KEY
            "datetime": PUN_timeseries_sell[i, 0],  # Timestamp for the entry

            # PUN VALUES KEY
            "PUN": PUN_timeseries_sell[i, 1],   # PUN value in kWh
            "soc": SoC[i] * 100,  # State of charge
            "c_d_energy": main.charged_energy[i] + main.discharged_energy[i],  # Total charged/discharged energy from BESS
            "Nominal C-rate": power_energy,  # Nominal charge rate
            "C-rate": (main.charged_energy[i] - main.discharged_energy[i]) / size,  # Actual C-rate
            "revenues": revenues[i],  # Total revenues for the time step
            "rev_BESS": -main.discharged_energy[i] * PUN_timeseries_sell[i, 1] / 1000,  # Revenue from BESS
            "rev_PV": main.discharged_from_pv[i] * PUN_timeseries_sell[i, 1] / 1000,  # Revenue from PV
            "rev_SC": float(main.load_self_consumption[i]) * PUN_timeseries_buy[i, 1] / 1000,  # Revenue from self-consumption
            "technology": technology,  # Technology used
            "size": size,  # Size of the BESS
            "dod": range_str,  # Depth of discharge
            "n_cycles": main.n_cycler[i],  # Number of cycles
            "energy_charged_from_PV_to_BESS": main.taken_from_pv[i],  # Energy charged from PV
            "energy_taken_from_grid_to_BESS": main.taken_from_grid[i],  # Energy taken from the grid
            "energy_sold_from_PV": main.discharged_from_pv[i],  # Energy sold from PV
            "energy_sold_from_BESS": -main.discharged_energy[i],  # Energy sold from BESS
            "energy_from_BESS_to_load": -main.from_BESS_to_load[i],  # Energy from BESS to load
            "energy_from_PV_to_load": -main.from_pv_to_load[i],  # Energy from PV to load
            "energy_self_consumed": main.load_self_consumption[i],  # Energy self consumed (load)
            "electrical_load": main.load[i],
            "pv_production": main.pv_production[i]

        }

        data.append(entry)  # Append entry to data list

    json_file_path = output_json_path  # Path for output JSON file
    with open(json_file_path, 'w') as json_file:
        json.dump(data, json_file, indent=4)  # Write data to JSON file with indentation

    import pandas as pd

    df = pd.DataFrame(data)

    # Excel file path
    excel_file_path = r'C:\Users\lorenzo.giannuzzo\PycharmProjects\BESS-Optimization\data\output\xlsx\results.xlsx' # Define your output path for the Excel file
    df.to_excel(excel_file_path, index=False)

    from argparser_l import weekends, args2

    # OLD FLAG NOT INFLUENCING IN THE CURRENT STATE OF THE CODE
    if weekends == 'True':
        subprocess.run(args2)