import multiprocessing
import numpy as np
import json

from pymoo.core.problem import StarmapParallelization
from multiprocessing import Pool, cpu_count
from objective_function import Revenues
from configuration import pop_size, soc_0, time_window, plot
from BESS_model import charge_rate_interpolated_func, discharge_rate_interpolated_func, size, charge_rate, discharge_rate, technology
from Economic_parameters import PUN_timeseries, time_window
from Optimizer import Optimizer
from argparser import output_json_path, range_str, minimize_C, soc_min, soc_max
from Plots import EnergyPlots
from PV import pv_production


__author__ = "Lorenzo Giannuzzo"
__maintainer__ = "Lorenzo Giannuzzo"
__email__ = "lorenzo.giannuzzo@polito.it"
__status__ = "in progress"
__version__ = "v0.2.1"
__license__ = "MIT"
__credits__ = ("Lorenzo Giannuzzo: Creation, Maintenance, Development, Idealization"
               "Daniele Salvatore Schiera: Idealization, Supervision"
               "Marco Massano: Idealization, Supervision")

# MAIN class creation

class Main:
    def __init__(self, multiprocessing=True):

        """
        Initializes the main object. Creates an instance of the objective function (Revenues) and the optimizer
        (Optimizer).
        """

        self.multiprocessing = multiprocessing

        if self.multiprocessing:

            n_processes = cpu_count() - 1
            self.pool = Pool(processes=n_processes)
            runner = StarmapParallelization(self.pool.starmap)
            self.objective_function = Revenues(elementwise_runner=runner,elementwise=True)

        else:

            self.objective_function = Revenues()

        self.optimizer = Optimizer(objective_function=self.objective_function, pop_size=pop_size, multiprocessing=multiprocessing)

    def run_optimization(self):

        """
        Executes the optimization, applies physical constraints, calculates revenues, and generates plots.
        """

        # Execute the optimization and get the solution

        solution = self.optimizer.maximize_revenues()

        if multiprocessing:
            self.pool.close()

        self.history = solution.history

        # Get the charge/discharge time series from the solution

        if minimize_C:

            # PARETO FRONT HANDLER

            of_values = np.array(solution.F[:,0])
            max_revenue_index = np.argmin(of_values)
            c_d_timeseries = solution.X[max_revenue_index, :time_window]
            alpha = solution.X[max_revenue_index, time_window:time_window * 2]
            alpha_mean = np.mean(alpha)

            print("\nAverage C/D reduction factor [%]:\n\n",alpha_mean*100)
            print("\nN Solutions in the Pareto-Front:\n\n",of_values.shape)

        else:
            alpha = np.ones(time_window)
            c_d_timeseries = solution.X

        # Apply physical constraints to the charge/discharge time series

        soc, charged_energy, discharged_energy, c_d_timeseries, taken_from_grid, taken_from_pv = self.apply_physical_constraints(c_d_timeseries,
                                                                                                     alpha)
        self.c_d_timeseries_final = c_d_timeseries
        self.soc = soc
        self.charged_energy = charged_energy
        self.discharged_energy = discharged_energy
        self.alpha = alpha
        self.taken_from_grid = taken_from_grid
        self.taken_from_pv = taken_from_pv
        self.discharged_from_pv = np.minimum(-pv_production['P'] + self.taken_from_pv , 0.0)

        # Calculate and print revenues

        self.calculate_and_print_revenues(charged_energy, discharged_energy, self.taken_from_grid, self.discharged_from_pv)

        # Generate plots of the results

        if plot:

            self.plot_results(soc, charged_energy, discharged_energy, np.abs(c_d_timeseries), PUN_timeseries[:,1],taken_from_grid,taken_from_pv)


    def apply_physical_constraints(self, c_d_timeseries,alpha):

        """
        Applies the physical constraints of the BESS to the charge/discharge time series.

        Args:
            c_d_timeseries (list): Charge/discharge time series.

        Returns:
            tuple: State of charge (SoC), charged energy, and discharged energy for each time step.
        """

        soc = [0.0] * time_window  # Initialize state of charge
        charged_energy = [0.0] * time_window  # Initialize charged energy
        discharged_energy = [0.0] * time_window  # Initialize discharged energy
        taken_from_grid = [0.0] * time_window  # Initialize energy taken from the grid
        taken_from_pv = [0.0] * time_window  # Initialize energy taken from the pv used to charged the BESS
        soc[0] = soc_0  # Initial state of charge

        c_func = charge_rate_interpolated_func  # Charge rate function
        d_func = discharge_rate_interpolated_func  # Discharge rate function

        for index in range(time_window - 1):

            if c_d_timeseries[index] >= 0:

                # Limit charge based on charge capacity and SoC

                c_d_timeseries[index] = min(c_d_timeseries[index]*alpha[index], min(c_func(soc[index])*alpha[index], soc_max - soc[index]))

            else:

                # Limit discharge based on discharge capacity and SoC

                c_d_timeseries[index] = max(c_d_timeseries[index]*alpha[index], max(-d_func(soc[index])*alpha[index], - soc[index] + soc_min))

            if c_d_timeseries[index] >= 0:

                # Calculate charged energy

                charged_energy[index] = c_d_timeseries[index] * size
                taken_from_grid[index] = np.maximum(charged_energy[index]-pv_production['P'].iloc[index], 0.0)
                taken_from_pv[index] = charged_energy[index] - taken_from_grid[index]

            else:

                # Calculate discharged energy

                discharged_energy[index] = c_d_timeseries[index] * size

            # Update SoC for the next time step

            if c_d_timeseries[index] >= 0:
                soc[index + 1] = min(soc_max, soc[index] + charged_energy[index]/size)

            else:
                soc[index + 1] = max(soc_min, soc[index] + discharged_energy[index]/size)

        return soc, charged_energy, discharged_energy, c_d_timeseries, taken_from_grid, taken_from_pv

    def calculate_and_print_revenues(self, charged_energy, discharged_energy, taken_from_grid, discharged_from_pv):

        """
        Calculates and prints total revenues for the optimized period.

        Args:
            charged_energy (list): Charged energy for each time step.
            discharged_energy (list): Discharged energy for each time step.

        """

        PUN_ts = PUN_timeseries[:,1]  # Time series of energy prices

        # Calculate revenues by summing the costs of charging and discharging

        rev = - (np.array(discharged_energy) * PUN_ts / 1000) - (taken_from_grid * PUN_ts / 1000) - discharged_from_pv * PUN_ts / 1000
        self.rev = rev
        # Print total revenues

        print("\nRevenues for optimized time window [EUROs]:\n\n", rev.sum())


    def plot_results(self, soc, charged_energy, discharged_energy, c_d_energy, PUN_Timeseries, taken_from_grid, taken_from_pv):

        """
        Generates plots of the state of charge, charged energy, discharged energy, and energy prices.

        Args:
            soc (list): State of charge for each time step.
            charged_energy (list): Charged energy for each time step.
            discharged_energy (list): Discharged energy for each time step.

        """
        if plot:

            plots = EnergyPlots(time_window, soc, charged_energy, discharged_energy, PUN_timeseries[:,1],taken_from_grid,taken_from_pv, pv_production['P'])
            plots.plot_soc()
            plots.plot_charged_energy()
            plots.plot_discharged_energy()
            plots.plot_combined_energy_with_pun(num_values=time_window)
            plots.plot_alpha_vs_timewindow(time_window, np.abs(c_d_energy), PUN_Timeseries, np.abs(c_d_energy)/self.alpha)


# MAIN EXECUTION

if __name__ == "__main__":

    # Create an instance of the Main class

    main = Main(multiprocessing=True)

    # Execute the optimization

    main.run_optimization()

    # POSTPROCESSING

    # Algorithm convergence

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
        EnergyPlots.PUN_plot(PUN_timeseries[:,1]) #
        EnergyPlots.convergence(len(main.history),time_window, pop_size, X, Y)
        EnergyPlots.c_d_plot(charge_rate, discharge_rate, charge_rate_interpolated_func, discharge_rate_interpolated_func)
        EnergyPlots.total_convergence(len(main.history), time_window, pop_size, X, Y)

    SoC = main.soc
    c_d_energy = main.c_d_timeseries_final
    alpha = main.alpha
    revenues = main.rev
    data = []

    # OUTPUT CREATION

    for i in range(len(PUN_timeseries[:,1])):
        entry = {
            "datetime": PUN_timeseries[i, 0].isoformat() + "Z",
            "PUN": PUN_timeseries[i, 1]/1000,
            "soc": SoC[i],
            "c_d_energy": c_d_energy[i]*size,
            "Nominal C-rate": np.abs(c_d_energy[i])/alpha[i],
            "C-rate": abs(c_d_energy[i]),
            "revenues": revenues[i],
            "rev_BESS": -main.discharged_energy[i] * PUN_timeseries[i,1] / 1000,
            "rev_PV": (pv_production['P'].iloc[i] - main.taken_from_pv[i])*PUN_timeseries[i,1]/1000,
            "technology": technology,
            "size": size,
            "dod": range_str,
            "energy_charged_from_PV": main.taken_from_pv[i],
            "energy_taken_from_grid": main.taken_from_grid[i],
            "energy_sold_from_PV": pv_production['P'].iloc[i] - main.taken_from_pv[i],
            "energy_sold_from_BESS": -main.discharged_energy[i]

        }

        data.append(entry)
        json_file_path = output_json_path
        with open(json_file_path, 'w') as json_file:
            json.dump(data, json_file, indent=4)




