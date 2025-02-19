""" BESS Optimization using NSGA-III Algorithm

    __author__ = "Lorenzo Giannuzzo"
    __maintainer__ = "Lorenzo Giannuzzo"
    __email__ = "lorenzo.giannuzzo@polito.it"
    __status__ = "in progress"
    __version__ = "v0.2.1"
    __license__ = "MIT"

Last Update of current code: 19/02/2025 """


# IMPORTING LIBRARIES AND MODULES FROM PROJECT
import numpy as np  # Numerical operations
import json  # JSON file handling
from pymoo.core.problem import StarmapParallelization  # Parallelization for optimization
from multiprocessing import Pool, cpu_count  # Multiprocessing utilities
from objective_function_s import Revenues  # Objective function for revenue calculation
from configuration_s import pop_size, soc_0, time_window, plot  # Configuration parameters
from BESS_model_s import charge_rate_interpolated_func, discharge_rate_interpolated_func, size, technology, BESS_model  # BESS model functions and parameters
from Economic_parameters_s import PUN_timeseries  # Economic parameters for pricing
from Optimizer_s import Optimizer  # Optimization class
from argparser_s import output_json_path, range_str, soc_min, power_energy, POD_power  # Argument parser parameters
from Plots_s import EnergyPlots  # Plotting utilities
from PV_s import pv_production  # Photovoltaic production data
import subprocess


# ----------------------------------------------------------------------------------------------------------------------

# CLASSES

# ----------------------------------------------------------------------------------------------------------------------

# CREATION OF CLASS MAIN
class Main:

    # CLASS ATTRIBUTES DEFINITION
    def __init__(self, multiprocessing=True):

        # DEFINE MULTIPROCESSING (TRUE OR FALSE)
        self.multiprocessing = multiprocessing

        # INITIALIZE MULTIPROCESSING
        if self.multiprocessing:
            n_processes = cpu_count() - 1  # Number of processes to use
            self.pool = Pool(processes=n_processes)  # Create a pool of worker processes
            runner = StarmapParallelization(self.pool.starmap)  # Set up parallelization runner
            self.objective_function = Revenues(elementwise_runner=runner, elementwise=True)  # Initialize objective
            # function for revenues
        else:
            self.objective_function = Revenues()  # Initialize without parallelization

        # SET OPTIMIZER
        self.optimizer = Optimizer(objective_function=self.objective_function, pop_size=pop_size,
                                   multiprocessing=multiprocessing)  # Initialize optimizer with the objective function

    # ------------------------------------------------------------------------------------------------------------------

    # FUNCTIONS

    # ------------------------------------------------------------------------------------------------------------------

    # EXECUTE OPTIMIZATION RUN FUNCTION
    def run_optimization(self):

        # GET SOLUTION FROM OPTIMIZATION TASK
        solution = self.optimizer.maximize_revenues()  # Run optimization to maximize revenues

        # CLOSE POOL IF MULTIPROCESSING IS TRUE
        if self.multiprocessing:
            self.pool.close()  # Close the pool of processes

        # SAVE OPTIMIZATION HISTORY
        if plot:
            self.history = solution.history  # Store optimization history for plotting

        # GET CHARGED/DISCHARGED ENERGY FROM SOLUTION
        c_d_timeseries = solution.X[:time_window]  # Extract charge/discharge time series from solution
        load_decision = [0.0] * time_window

        # APPLY PHYSICAL CONSTRAINTS
        (soc, charged_energy, discharged_energy, c_d_timeseries, taken_from_grid, discharged_from_pv,
         taken_from_pv, n_cycler, load_self_consumption, from_pv_to_load, from_BESS_to_load, shared_energy_bess) = \
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
        self.shared_energy_bess = shared_energy_bess

        # GET LOAD DATA
        from Load import data

        # CALCULATE AND PRINT REVENUES
        self.calculate_and_print_revenues(self.charged_energy, self.discharged_energy, self.taken_from_grid,
                                          self.discharged_from_pv, self.from_pv_to_load, self.from_BESS_to_load,
                                          shared_energy_bess)
        # Calculate and display revenues

        # GENERATE PLOTS IF PLOT FLAG IS TRUE
        if plot:
            self.plot_results(soc, charged_energy, discharged_energy, c_d_timeseries, PUN_timeseries[:, 1],
                              taken_from_grid, taken_from_pv, discharged_from_pv, load_self_consumption,
                              from_pv_to_load,
                              from_BESS_to_load, shared_energy_bess, data)  # Generate plots for the results

    # CONSTRAINTS FUNCTION DEFINITION
    @staticmethod
    def apply_physical_constraints(c_d_timeseries, load_decision):

        from Load import data
        from BESS_model_s import degradation
        from argparser_s import n_cycles
        from argparser_s import soc_max, soc_min

        # GET CHARGE/DISCHARGE INTERPOLATED FUNCTIONS
        c_func = charge_rate_interpolated_func
        d_func = discharge_rate_interpolated_func

        # INITIALIZE FIRST PARAMETERS
        soc = [0.0] * time_window
        soc[0] = soc_0
        bess_model = BESS_model(time_window, PUN_timeseries, soc, size, c_func, d_func)  # Create BESS model instance

        charged_energy, discharged_energy = bess_model.run_simulation(c_d_timeseries)
        # Run simulation to get energy values

        # GET VARIABLES
        production = pv_production['P']
        rec_load = data

        # INITIALIZE VARIABLES
        n_cycler = [0] * time_window
        n_cycler[0] = n_cycles
        # total_available_energy = [0.0] * time_window
        # self_consumption = [0.0] * time_window
        from_pv_to_load = [0.0] * time_window
        from_BESS_to_load = [0.0] * time_window
        load_self_consumption = [0.0] * time_window
        taken_from_pv = [0.0] * time_window
        charged_energy_from_grid_to_BESS = [0.0] * time_window
        discharged_from_pv = [0.0] * time_window
        remaining_production = [0.0] * time_window
        shared_energy_rec = [0.0] * time_window
        shared_energy_bess = [0.0] * time_window

        # Loop through time window to calculate state of charge and cycles
        for i in range(time_window - 1):

            # UPDATE SOC MAX BESED ON ITS ACTUAL AND PAST DEGRADATION
            # GET PREVIOUS NUMBER OF CYCLES
            n_cycles_prev = n_cycles
            max_capacity = degradation(n_cycles_prev) / 100
            soc_max = min(soc_max, max_capacity)

            # EVALUATE THE ENERGY USED TO CHARGE THE BESS TAKEN FROM THE GRID (IF CHARGED_ENERGY_FROM_BESS IS NEGATIVE,
            # MEANING THAT THE BESS IS DISCHARGING, THIS VALUE IS = 0
            charged_energy_from_grid_to_BESS[i] = charged_energy[i]

            # UPDATE THE ENERGY THAT THE BESS WANT TO CHARGED AS SUM OF THE ONE CHARGED FROM GRID TO BESS AND THE ENERGY
            # TAKEN FROM PV TO THE BESS
            charged_energy[i] = charged_energy_from_grid_to_BESS[i]

            # UPDATE THE ENERGY DISCHARGED FROM PV DIRECTLY TO THE GRID REDUCING ITS ORIGINAL VALUE BY THE ONE THAT
            # GOES FROM PV TO THE BESS AND FROM THE PV TO THE LOAD
            discharged_from_pv[i] = -production[i]  # NEGATIVE VALUE

            # EVALUATE THE ENERGY DISCHARGED FROM BESS BASED TO THE VALUE OF THE OTHER OBTAINED ENERGY VECTORS
            discharged_energy[i] = np.maximum(np.maximum(discharged_energy[i],
                                                         -(soc[i] - soc_min) * size * power_energy),
                                              -size * power_energy)

            # APPLY POD CONSTRAINTS TO ENERGY VECTORS
            if charged_energy_from_grid_to_BESS[i] > POD_power:
                charged_energy_from_grid_to_BESS[i] = np.maximum(POD_power, 0.0)

            if np.abs(discharged_energy[i]) > POD_power:
                discharged_energy[i] = np.minimum(-POD_power, discharged_energy[i])

            charged_energy_from_grid_to_BESS[i] = charged_energy[i]

            # UPDATE THE ENERGY THAT THE BESS WANT TO CHARGED AS SUM OF THE ONE CHARGED FROM GRID TO BESS AND THE ENERGY
            # TAKEN FROM PV TO THE BESS
            charged_energy[i] = charged_energy_from_grid_to_BESS[i]

            # UPDATE THE ENERGY DISCHARGED FROM PV DIRECTLY TO THE GRID REDUCING ITS ORIGINAL VALUE BY THE ONE THAT
            # GOES FROM PV TO THE BESS AND FROM THE PV TO THE LOAD
            discharged_from_pv[i] = np.minimum(-production[i], 0.0)
            discharged_energy[i] = np.maximum(np.maximum(discharged_energy[i], -(soc[i] - soc_min) * size *
                                                         power_energy), -size * power_energy)

            # UPDATE SOC
            # IF BESS I CHARGING
            if c_d_timeseries[i] >= 0:
                soc[i + 1] = min(soc_max, soc[i] + (charged_energy[i]) / size)

            # IF BESS IS DISCHARGING
            else:
                soc[i + 1] = max(soc_min, soc[i] + (discharged_energy[i]) / size)

            # EVALUATING SHARED ENERGY
            shared_energy_rec[i] = np.minimum(rec_load[i], np.abs(discharged_from_pv[i]))
            remaining_production[i] = np.maximum(np.abs(discharged_from_pv[i]) - shared_energy_rec[i], 0.0)
            shared_energy_bess[i] = np.minimum(remaining_production[i], charged_energy[i])

            # UPDATING NUMBER OF CYCLES
            total_energy = charged_energy[i] + np.abs(discharged_energy[i])
            actual_capacity = size * degradation(n_cycles_prev) / 100
            n_cycles = n_cycles_prev + total_energy / actual_capacity

        # EVALUATE THE NUMBER OF CYCLES DONE BY BESS
        total_charged = np.sum(charged_energy)
        total_discharged = np.sum(-discharged_energy)
        total_energy = total_charged + total_discharged

        from argparser_s import n_cycles
        n_cycles_prev = n_cycles
        actual_capacity = size * degradation(n_cycles_prev) / 100
        n_cycles = total_energy / actual_capacity

        return (soc, charged_energy, discharged_energy, c_d_timeseries, charged_energy_from_grid_to_BESS,
                discharged_from_pv, taken_from_pv, n_cycler, load_self_consumption, from_pv_to_load, from_BESS_to_load,
                shared_energy_bess)

    # CALCULATE AND PRINT REVENUES FUNCTION
    def calculate_and_print_revenues(self, charged_energy, discharged_energy, taken_from_grid, discharged_from_pv,
                                     from_pv_to_load, from_BESS_to_load, shared_energy_bess):

        # GET PUN VALUES
        PUN_ts = PUN_timeseries[:, 1]

        # EVALUATE REVENUES
        rev = np.array(np.abs(discharged_energy) * PUN_ts / 1000
                       - np.abs(taken_from_grid * PUN_ts * 1.1 / 1000)
                       + np.array(shared_energy_bess) * 120 / 1000
                       )

        self.rev = rev

        # EVALUATES TYPICAL DAYS REVENUES
        num_weeks = 12  # Number of weeks to evaluate
        hours_per_week = 24  # Hours per week
        revenues_weeks = np.zeros(num_weeks)  # Initialize weekly revenues array

        # EVALUATE REVENUES
        for i in range(num_weeks):
            start = i * hours_per_week  # Start index for the week
            end = start + hours_per_week  # End index for the week
            revenues_weeks[i] = np.sum(rev[start:end]) * 30  # Calculate weekly revenues

        revenues_final = revenues_weeks

        # EVALUATE TOTAL REVENUES
        rev = np.sum(revenues_final)

        # DISPLAY TOTAL REVENUES
        print("\nRevenues for optimized time window [EUROs]:\n\n", rev.sum())

    # DEFINE PLOT RESULTS FUNCTION
    def plot_results(self, soc, charged_energy, discharged_energy, c_d_energy, PUN_Timeseries, taken_from_grid,
                     taken_from_pv, discharged_from_pv, self_consumption, from_pv_to_load, from_BESS_to_load,
                     shared_energy_bess, load):

        # IMPORT LOAD DATA
        from Load import data

        # EXECUTE PLOTS
        if plot:
            plots = EnergyPlots(time_window, soc, charged_energy, discharged_energy, PUN_timeseries[:, 1],
                                taken_from_grid, taken_from_pv, pv_production['P'], discharged_from_pv,
                                self_consumption, from_pv_to_load, from_BESS_to_load, shared_energy_bess,
                                np.array(data))

            # EXECUTE PLOT FUNCTIONS
            plots.Total_View(num_values=time_window)
            plots.plot_daily_energy_flows(num_values=time_window)
            plots.plot_degradation()


# ----------------------------------------------------------------------------------------------------------------------

# MAIN EXECUTION

# ----------------------------------------------------------------------------------------------------------------------


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
    for i in range(len(PUN_timeseries[:, 1])):
        entry = {
            # DATETIME KEY
            "datetime": PUN_timeseries[i, 0],  # Timestamp for the entry

            # PUN VALUES KEY
            "PUN": PUN_timeseries[i, 1] / 1000,  # PUN value in kWh
            "soc": SoC[i],  # State of charge
            "c_d_energy": main.charged_energy[i] + main.discharged_energy[i],  # Total charged/discharged energy from BESS
            "Nominal C-rate": power_energy,  # Nominal charge rate
            "C-rate": (main.charged_energy[i] - main.discharged_energy[i]) / size,  # Actual C-rate
            "revenues": revenues[i],  # Total revenues for the time step
            "rev_BESS": -main.discharged_energy[i] * PUN_timeseries[i, 1] / 1000,  # Revenue from BESS
            "rev_PV": main.discharged_from_pv[i] * PUN_timeseries[i, 1] / 1000,  # Revenue from PV
            "rev_SC": float(main.load_self_consumption[i]) * PUN_timeseries[i, 1] / 1000,  # Revenue from self-consumption
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
            "energy_self_consumed": main.load_self_consumption[i]  # Energy self consumed (load)

        }

        # Append entry to data list
        data.append(entry)

    json_file_path = output_json_path  # Path for output JSON file
    with open(json_file_path, 'w') as json_file:
        json.dump(data, json_file, indent=4)  # Write data to JSON file with indentation

    from argparser_s import weekends, args2

    # OLD FLAG NOT INFLUENCING IN THE CURRENT STATE OF THE CODE
    if weekends == 'True':
        subprocess.run(args2)
