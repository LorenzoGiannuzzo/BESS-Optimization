"""

BESS Optimization using NSGA-III Algorithm

    __author__ = "Lorenzo Giannuzzo"
    __maintainer__ = "Lorenzo Giannuzzo"
    __email__ = "lorenzo.giannuzzo@polito.it"
    __status__ = "in progress"
    __version__ = "v0.2.1"
    __license__ = "MIT"

Last Update of current code: 20/11/2024 - 17:18

"""

# IMPORTING LIBRARIES AND MODULES FROM PROJECT

import numpy as np
import json
from pymoo.core.problem import StarmapParallelization
from multiprocessing import Pool, cpu_count
from objective_function_s import Revenues
from configuration_s import pop_size, soc_0, time_window, plot
from BESS_model_s import charge_rate_interpolated_func, discharge_rate_interpolated_func, size, technology, BESS_model
from Economic_parameters_s import PUN_timeseries
from Optimizer_s import Optimizer
from argparser_s import output_json_path, range_str, soc_min, power_energy, POD_power
from Plots_s import EnergyPlots
from PV_s import pv_production


# CREATION OF CLASS MAIN
class Main:
    def __init__(self, multiprocessing=True):
        """
        Initializes the main object. Creates an instance of the objective function (Revenues) and the optimizer
        (Optimizer).
        """
        self.multiprocessing = multiprocessing

        # INITIALIZE MULTIPROCESSING
        if self.multiprocessing:
            n_processes = cpu_count() - 1
            self.pool = Pool(processes=n_processes)
            runner = StarmapParallelization(self.pool.starmap)
            # SET OBJECTIVE FUNCTION
            self.objective_function = Revenues(elementwise_runner=runner, elementwise=True)
        else:
            # SET OBJECTIVE FUNCTION
            self.objective_function = Revenues()

        # SET OPTIMIZER
        self.optimizer = Optimizer(objective_function=self.objective_function, pop_size=pop_size,
                                   multiprocessing=multiprocessing)

    # DEFINE RUN OPTIMIZATION FUNCTION
    def run_optimization(self):
        """
        Executes the optimization, applies physical constraints, calculates revenues, and generates plots.
        """
        # GET SOLUTION FROM OPTIMIZATION TASK
        solution = self.optimizer.maximize_revenues()

        if self.multiprocessing:
            self.pool.close()

        # SAVE OPTIMIZATION HISTORY
        self.history = solution.history

        # GET CHARGED/DISCHARGED ENERGY FROM SOLUTION
        c_d_timeseries = solution.X

        # APPLY PHYSICAL CONSTRAINTS
        (soc, charged_energy, discharged_energy, c_d_timeseries, taken_from_grid, discharged_from_pv,
         taken_from_pv, n_cycler) = self.apply_physical_constraints(c_d_timeseries)

        self.c_d_timeseries_final = c_d_timeseries
        self.soc = soc
        self.charged_energy = charged_energy
        self.discharged_energy = discharged_energy
        self.taken_from_grid = taken_from_grid
        self.taken_from_pv = taken_from_pv
        self.discharged_from_pv = discharged_from_pv
        self.n_cycler = n_cycler

        # Calculate and print revenues
        self.calculate_and_print_revenues(charged_energy, discharged_energy, self.taken_from_grid,
                                          self.discharged_from_pv)

        # Generate plots of the results
        if plot:
            self.plot_results(soc, charged_energy, discharged_energy, np.abs(c_d_timeseries), PUN_timeseries[:, 1],
                              taken_from_grid, taken_from_pv, discharged_from_pv)

    @staticmethod
    def apply_physical_constraints(c_d_timeseries):
        c_func = charge_rate_interpolated_func  # Charge rate function
        d_func = discharge_rate_interpolated_func  # Discharge rate function
        soc = [0.0] * time_window  # Initialize state of charge
        soc[0] = soc_0  # Initial state of charge

        bess_model = BESS_model(time_window, PUN_timeseries, soc, size, c_func, d_func)
        charged_energy, discharged_energy = bess_model.run_simulation(c_d_timeseries)

        taken_from_pv = np.minimum(charged_energy, pv_production['P'])
        charged_energy_grid = np.maximum(charged_energy - taken_from_pv, 0.0)
        discharged_from_pv = np.minimum(-pv_production['P'] + taken_from_pv, 0.0)

        for i in range(len(discharged_from_pv)):
            if -discharged_from_pv[i] - discharged_energy[i] > POD_power:
                discharged_from_pv[i] = -min(POD_power, -discharged_from_pv[i])
                discharged_energy[i] = -min(POD_power - abs(discharged_from_pv[i]), -discharged_energy[i])

            if charged_energy_grid[i] >= POD_power:
                charged_energy_grid[i] = min(charged_energy_grid[i], POD_power)
                charged_energy[i] = charged_energy_grid[i] + taken_from_pv[i]

        from argparser_s import n_cycles

        n_cycler = [0] * time_window
        n_cycler[0] = n_cycles

        for index in range(time_window - 1):
            from BESS_model_s import degradation
            from argparser_s import soc_max

            n_cycles_prev = n_cycles
            max_capacity = degradation(n_cycles_prev) / 100
            soc_max = min(soc_max, max_capacity)

            if c_d_timeseries[index] >= 0:
                soc[index + 1] = min(soc_max, soc[index] + charged_energy[index] / size)
                charged_energy[index] = (soc[index + 1] - soc[index]) * size
            else:
                soc[index + 1] = max(soc_min, soc[index] + discharged_energy[index] / size)
                discharged_energy[index] = (soc[index + 1] - soc[index]) * size

            total_energy = charged_energy[index] + np.abs(discharged_energy[index])
            actual_capacity = size * degradation(n_cycles_prev) / 100
            n_cycles = n_cycles_prev + total_energy / actual_capacity
            n_cycler[index+1] = n_cycles

        return (soc, charged_energy, discharged_energy, c_d_timeseries, charged_energy_grid,
                discharged_from_pv, taken_from_pv, n_cycler)

    def calculate_and_print_revenues(self, charged_energy, discharged_energy, taken_from_grid, discharged_from_pv):
        """
        Calculates and prints total revenues for the optimized period.
        Args:
            charged_energy (list): Charged energy for each time step.
            discharged_energy (list): Discharged energy for each time step.
        """
        PUN_ts = PUN_timeseries[:, 1]  # Time series of energy prices
        rev = (- (np.array(discharged_energy) * PUN_ts / 1000) - (taken_from_grid * PUN_ts / 1000) -
               discharged_from_pv * PUN_ts / 1000)

        self.rev = rev

        # Calcolare le revenues settimanali
        num_settimane = 12
        ore_per_settimana = 24

        # Inizializza un array per le revenues settimanali
        revenues_settimanali = np.zeros(num_settimane)

        # Calcola le revenues per ogni settimana
        for i in range(num_settimane):
            inizio = i * ore_per_settimana
            fine = inizio + ore_per_settimana
            revenues_settimanali[i] = np.sum(rev[inizio:fine]) * 30

        # Moltiplica le revenues settimanali per 4
        revenues_finali = revenues_settimanali

        # Calcola la somma totale delle revenues finali
        revv = np.sum(revenues_finali)


        print("\nRevenues for optimized time window [EUROs]:\n\n", revv.sum())

    def plot_results(self, soc, charged_energy, discharged_energy, c_d_energy, PUN_Timeseries, taken_from_grid,
                     taken_from_pv, discharged_from_pv):
        """
        Generates plots of the state of charge, charged energy, discharged energy, and energy prices.
        Args:
            soc (list): State of charge for each time step.
            charged_energy (list): Charged energy for each time step.
            discharged_energy (list): Discharged energy for each time step.
        """
        if plot:
            plots = EnergyPlots(time_window, soc, charged_energy, discharged_energy, PUN_timeseries[:, 1],
                                taken_from_grid, taken_from_pv, pv_production['P'], discharged_from_pv)
            plots.plot_combined_energy_with_pun(num_values=time_window)
            plots.Total_View(num_values=time_window)
            plots.plot_daily_energy_flows(num_values=time_window)
            #if plot_monthly == True:
               # plots.Total_View_Monthly(time_window)
                #plots.Total_View_Monthly_SoC(time_window)

            plots.PV_View(num_values=time_window)
            plots.POD_View(num_values=time_window)
            plots.plot_alpha_vs_timewindow(time_window, (charged_energy - discharged_energy) / size, PUN_Timeseries,
                                           [power_energy] * (time_window))
            plots.Total_View_cycles(time_window, main.n_cycler)
            plots.plot_degradation()


# MAIN EXECUTION
if __name__ == "__main__":
    main = Main(multiprocessing=True)
    main.run_optimization()

    # POSTPROCESSING
    X = []
    Y = []

    for j in range(len(main.history)):
        X.append([])
        for i in range(pop_size):
            X[j].append(main.history[j].pop[i].get('X'))

    for j in range(len(main.history)):
        Y.append([])
        for i in range(pop_size):
            Y[j].append(main.history[j].pop[i].get('f'))

    X = np.array(X)
    Y = np.array(Y)

    # PLOTS
    if plot:
        #EnergyPlots.PUN_plot(PUN_timeseries[:, 1])
        #EnergyPlots.convergence(len(main.history), time_window, pop_size, X, Y)
        #EnergyPlots.c_d_plot(charge_rate, discharge_rate, charge_rate_interpolated_func,
                              #discharge_rate_interpolated_func)
        EnergyPlots.total_convergence(len(main.history), time_window, pop_size, X, Y)



    SoC = main.soc
    c_d_energy = main.c_d_timeseries_final
    revenues = main.rev
    data = []


    # OUTPUT CREATION
    for i in range(len(PUN_timeseries[:, 1])):
        entry = {

            "datetime": PUN_timeseries[i, 0],
            "PUN": PUN_timeseries[i, 1] / 1000,
            "soc": SoC[i],
            "c_d_energy": main.charged_energy[i] + main.discharged_energy[i],
            "Nominal C-rate": power_energy,
            "C-rate": (main.charged_energy[i] - main.discharged_energy[i]) / size,
            "revenues": revenues[i],
            "rev_BESS": -main.discharged_energy[i] * PUN_timeseries[i, 1] / 1000,
            "rev_PV": (pv_production['P'].iloc[i] - main.taken_from_pv[i]) * PUN_timeseries[i, 1] / 1000,
            "technology": technology,
            "size": size,
            "dod": range_str,
            "n_cycles": main.n_cycler[i],
            "energy_charged_from_PV": main.taken_from_pv[i],
            "energy_taken_from_grid": main.taken_from_grid[i],
            "energy_sold_from_PV": pv_production['P'].iloc[i] - main.taken_from_pv[i],
            "energy_sold_from_BESS": -main.discharged_energy[i]

        }
        data.append(entry)

    json_file_path = output_json_path
    with open(json_file_path, 'w') as json_file:
        json.dump(data, json_file, indent=4)




