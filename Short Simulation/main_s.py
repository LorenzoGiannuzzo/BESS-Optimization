"""

BESS Optimization using NSGA-III Algorithm

    __author__ = "Lorenzo Giannuzzo"
    __maintainer__ = "Lorenzo Giannuzzo"
    __email__ = "lorenzo.giannuzzo@polito.it"
    __status__ = "in progress"
    __version__ = "v0.2.1"
    __license__ = "MIT"

Last Update of current code: 09/01/2025 - 17:18

"""

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

            # SET OBJECTIVE FUNCTION
            self.objective_function = Revenues(elementwise_runner=runner, elementwise=True)  # Initialize objective function for revenues
        else:
            # SET OBJECTIVE FUNCTION
            self.objective_function = Revenues()  # Initialize without parallelization

        # SET OPTIMIZER
        self.optimizer = Optimizer(objective_function=self.objective_function, pop_size=pop_size,
                                   multiprocessing=multiprocessing)  # Initialize optimizer with the objective function

    # DEFINE RUN OPTIMIZATION FUNCTION
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
        c_d_timeseries = solution.X  # Extract charge/discharge time series from solution

        # APPLY PHYSICAL CONSTRAINTS
        (soc, charged_energy, discharged_energy, c_d_timeseries, taken_from_grid, discharged_from_pv,
         taken_from_pv, n_cycler) = self.apply_physical_constraints(c_d_timeseries)  # Apply constraints to the solution

        # DEFINE CLASS ATTRIBUTES AS CONSTRAINED SOLUTION OBTAINED FORM OPTIMIZATION
        self.c_d_timeseries_final = c_d_timeseries  # Final charge/discharge time series
        self.soc = soc  # State of charge
        self.charged_energy = charged_energy  # Charged energy time series
        self.discharged_energy = discharged_energy  # Discharged energy time series
        self.taken_from_grid = taken_from_grid  # Energy taken from the grid
        self.taken_from_pv = taken_from_pv  # Energy taken from PV
        self.discharged_from_pv = discharged_from_pv  # Energy discharged from PV
        self.n_cycler = n_cycler  # Number of cycles

        # CALCULATE AND PRINT REVENUES
        self.calculate_and_print_revenues(charged_energy, discharged_energy, self.taken_from_grid,
                                          self.discharged_from_pv)  # Calculate and display revenues

        # GENERATE PLOTS IF PLOT FLAG IS TRUE
        if plot:
            self.plot_results(soc, charged_energy, discharged_energy, np.abs(c_d_timeseries), PUN_timeseries[:, 1],
                              taken_from_grid, taken_from_pv, discharged_from_pv)  # Generate plots for the results

    # CONSTRAINTS FUNCTION DEFINITION
    @staticmethod
    def apply_physical_constraints(c_d_timeseries):
        """
        Applies physical constraints to the charge/discharge time series.
        Args:
            c_d_timeseries (list): Time series of charge/discharge values.
        Returns:
            tuple: Contains state of charge, charged energy, discharged energy, and other relevant data.
        """
        c_func = charge_rate_interpolated_func  # Charge rate function
        d_func = discharge_rate_interpolated_func  # Discharge rate function
        soc = [0.0] * time_window  # Initialize state of charge
        soc[0] = soc_0  # Set initial state of charge

        bess_model = BESS_model(time_window, PUN_timeseries, soc, size, c_func, d_func)  # Create BESS model instance
        charged_energy, discharged_energy = bess_model.run_simulation(c_d_timeseries)  # Run simulation to get energy values

        taken_from_pv = np.minimum(charged_energy, pv_production['P'])  # Energy taken from PV
        charged_energy_grid = np.maximum(charged_energy - taken_from_pv, 0.0)  # Energy charged from the grid
        discharged_from_pv = np.minimum(-pv_production['P'] + taken_from_pv, 0.0)  # Energy discharged from PV

        # Loop through each time step to apply constraints
        for i in range(len(discharged_from_pv)):
            if -discharged_from_pv[i] - discharged_energy[i] > POD_power:  # Check for discharge limits
                discharged_from_pv[i] = -min(POD_power, -discharged_from_pv[i])  # Limit discharge from PV
                discharged_energy[i] = -min(POD_power - abs(discharged_from_pv[i]), -discharged_energy[i])  # Adjust discharged energy

            if charged_energy_grid[i] >= POD_power:  # Check for charge limits
                charged_energy_grid[i] = min(charged_energy_grid[i], POD_power)  # Limit charge from grid
                charged_energy[i] = charged_energy_grid[i] + taken_from_pv[i]  # Update charged energy

        from argparser_s import n_cycles  # Import number of cycles

        n_cycler = [0] * time_window  # Initialize number of cycles
        n_cycler[0] = n_cycles  # Set initial number of cycles

        # Loop through time window to calculate state of charge and cycles
        for index in range(time_window - 1):
            from BESS_model_s import degradation  # Import degradation function
            from argparser_s import soc_max  # Import maximum state of charge

            n_cycles_prev = n_cycles  # Store previous number of cycles
            max_capacity = degradation(n_cycles_prev) / 100  # Calculate maximum capacity based on degradation
            soc_max = min(soc_max, max_capacity)  # Limit state of charge to maximum capacity

            if c_d_timeseries[index] >= 0:  # If charging
                soc[index + 1] = min(soc_max, soc[index] + charged_energy[index] / size)  # Update state of charge
                charged_energy[index] = (soc[index + 1] - soc[index]) * size  # Calculate charged energy
            else:  # If discharging
                soc[index + 1] = max(soc_min, soc[index] + discharged_energy[index] / size)  # Update state of charge
                discharged_energy[index] = (soc[index + 1] - soc[index]) * size  # Calculate discharged energy

            total_energy = charged_energy[index] + np.abs(discharged_energy[index])  # Total energy calculation
            actual_capacity = size * degradation(n_cycles_prev) / 100  # Actual capacity based on degradation
            n_cycles = n_cycles_prev + total_energy / actual_capacity  # Update number of cycles
            n_cycler[index + 1] = n_cycles  # Store number of cycles for the next time step

        return (soc, charged_energy, discharged_energy, c_d_timeseries, charged_energy_grid,
                discharged_from_pv, taken_from_pv, n_cycler)  # Return all relevant data

    def calculate_and_print_revenues(self, charged_energy, discharged_energy, taken_from_grid, discharged_from_pv):
        """
        Calculates and prints total revenues for the optimized period.
        Args:
            charged_energy (list): Charged energy for each time step.
            discharged_energy (list): Discharged energy for each time step.
        """
        PUN_ts = PUN_timeseries[:, 1]  # Time series of energy prices
        rev = (- (np.array(discharged_energy) * PUN_ts / 1000) - (taken_from_grid * PUN_ts / 1000) -
               discharged_from_pv * PUN_ts / 1000)  # Calculate revenues from discharged energy and grid usage

        self.rev = rev  # Store calculated revenues

        # EVALUATES TYPICAL DAYS REVENUES
        num_settimane = 12  # Number of weeks to evaluate
        ore_per_settimana = 24  # Hours per week
        revenues_settimanali = np.zeros(num_settimane)  # Initialize weekly revenues array

        # EVALUATE REVENUES
        for i in range(num_settimane):
            inizio = i * ore_per_settimana  # Start index for the week
            fine = inizio + ore_per_settimana  # End index for the week
            revenues_settimanali[i] = np.sum(rev[inizio:fine]) * 30  # Calculate weekly revenues

        revenues_finali = revenues_settimanali  # Store final revenues

        # EVALUATE TOTAL REVENUES
        revv = np.sum(revenues_finali)  # Total revenues calculation

        # DISPLAY TOTAL REVENUES
        print("\nRevenues for optimized time window [EUROs]:\n\n", revv.sum())  # Print total revenues

    # DEFINE PLOT RESULTS FUNCTION
    def plot_results(self, soc, charged_energy, discharged_energy, c_d_energy, PUN_Timeseries, taken_from_grid,
                     taken_from_pv, discharged_from_pv):
        """
        Generates plots of the state of charge, charged energy, discharged energy, and energy prices.
        Args:
            soc (list): State of charge for each time step.
            charged_energy (list): Charged energy for each time step.
            discharged_energy (list): Discharged energy for each time step.
        """
        # EXECUTE PLOTS
        if plot:
            plots = EnergyPlots(time_window, soc, charged_energy, discharged_energy, PUN_timeseries[:, 1],
                                taken_from_grid, taken_from_pv, pv_production['P'], discharged_from_pv)  # Create EnergyPlots instance
            plots.plot_combined_energy_with_pun(num_values=time_window)  # Plot combined energy with price
            plots.Total_View(num_values=time_window)  # Total view plot
            plots.plot_daily_energy_flows(num_values=time_window)  # Daily energy flow plot
            plots.PV_View(num_values=time_window)  # PV view plot
            plots.POD_View(num_values=time_window)  # POD view plot
            plots.plot_alpha_vs_timewindow(time_window, (charged_energy - discharged_energy) / size, PUN_Timeseries,
                                           [power_energy] * (time_window))  # Plot alpha vs time window
            plots.Total_View_cycles(time_window, self.n_cycler)  # Total view of cycles
            plots.plot_degradation()  # Plot degradation

# MAIN EXECUTION
if __name__ == "__main__":



    main = Main(multiprocessing=True)  # Create Main instance with multiprocessing enabled
    main.run_optimization()  # Run optimization

    # POSTPROCESSING
    X = []  # Initialize list for optimization history X values
    Y = []  # Initialize list for optimization history Y values

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

    X = np.array(X)  # Convert X to numpy array
    Y = np.array(Y)  # Convert Y to numpy array

    # EXECUTE ADDITIONAL PLOTS IF PLOT FLAG IS TRUE
    if plot:
        EnergyPlots.total_convergence(len(main.history), time_window, pop_size, X, Y)  # Plot total convergence

    # GET SOC VALUES
    SoC = main.soc  # Retrieve state of charge values

    # GET CHARGED/DISCHARGED ENERGY VALUES
    c_d_energy = main.c_d_timeseries_final  # Retrieve final charge/discharge energy values

    # GET REVENUES VALUES
    revenues = main.rev  # Retrieve revenue values
    data = []  # Initialize list for output data

    # OUTPUT CREATION AS .JSON FILE
    for i in range(len(PUN_timeseries[:, 1])):
        entry = {
            # DATETIME KEY
            "datetime": PUN_timeseries[i, 0],  # Timestamp for the entry

            # PUN VALUES KEY
            "PUN": PUN_timeseries[i, 1] / 1000,  # PUN value in kWh
            "soc": SoC[i],  # State of charge
            "c_d_energy": main.charged_energy[i] + main.discharged_energy[i],  # Total charged/discharged energy
            "Nominal C-rate": power_energy,  # Nominal charge rate
            "C-rate": (main.charged_energy[i] - main.discharged_energy[i]) / size,  # Actual C-rate
            "revenues": revenues[i],  # Revenue for the time step
            "rev_BESS": -main.discharged_energy[i] * PUN_timeseries[i, 1] / 1000,  # Revenue from BESS
            "rev_PV": (pv_production['P'].iloc[i] - main.taken_from_pv[i]) * PUN_timeseries[i, 1] / 1000,  # Revenue from PV
            "technology": technology,  # Technology used
            "size": size,  # Size of the BESS
            "dod": range_str,  # Depth of discharge
            "n_cycles": main.n_cycler[i],  # Number of cycles
            "energy_charged_from_PV": main.taken_from_pv[i],  # Energy charged from PV
            "energy_taken_from_grid": main.taken_from_grid[i],  # Energy taken from the grid
            "energy_sold_from_PV": pv_production['P'].iloc[i] - main.taken_from_pv[i],  # Energy sold from PV
            "energy_sold_from_BESS": -main.discharged_energy[i]  # Energy sold from BESS
        }
        data.append(entry)  # Append entry to data list

    json_file_path = output_json_path  # Path for output JSON file
    with open(json_file_path, 'w') as json_file:
        json.dump(data, json_file, indent=4)  # Write data to JSON file with indentation

    from argparser_s import weekends, args2

    if weekends == 'True':
        subprocess.run(args2)
