import multiprocessing
import numpy as np
import json

from pymoo.core.problem import StarmapParallelization
from multiprocessing import Pool, cpu_count
from objective_function import Revenues
from configuration import pop_size, soc_0, time_window, plot
from BESS_model import charge_rate_interpolated_func, discharge_rate_interpolated_func, size, charge_rate, discharge_rate, technology
from Economic_parameters import PUN_timeseries
from Optimizer import Optimizer
from argparser import output_json_path, range_str
from Plots import EnergyPlots

from pymoo.decomposition.asf import ASF



class Main:
    def __init__(self, multiprocessing=True):

        """
        Initializes the main object. Creates an instance of the objective function (Revenues) and the optimizer
        (Optimizer).
        """

        self.multiprocessing = multiprocessing

        if self.multiprocessing:

            n_processes = cpu_count() - 1

            #print(n_processes)# Set the number of processes

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

        #c_d_timeseries = solution.X
        #c_d_timeseries = c_d_timeseries[13,:time_window]
        #alpha_mean = np.mean(solution.X[13,time_window:time_window*2])
        #alpha = solution.X[13,time_window:time_window*2]

        alpha_var = np.var(solution.X[:, time_window:time_window * 2], axis=1)

        # Trova l'indice della riga con la varianza più alta
        max_var_index = np.argmax(alpha_var)

        # Usa questo indice per selezionare le righe corrispondenti
        c_d_timeseries = solution.X[max_var_index, :time_window]
        alpha = solution.X[max_var_index, time_window:time_window * 2]
        alpha_mean = np.mean(alpha)

        #weights = np.array([0.5, 0.5])
        #decomp = ASF()
        #prova = decomp(solution.X, weights).argmin()

        #print(solution.X)
        #print(c_d_timeseries.shape)
        print("\nAverage C/D reduction factor [%]:\n\n",alpha_mean*100)
        #print(alpha)

        # Apply physical constraints to the charge/discharge time series

        soc, charged_energy, discharged_energy, c_d_timeseries = self.apply_physical_constraints(c_d_timeseries,alpha)

        self.c_d_timeseries_final = c_d_timeseries
        self.soc = soc
        self.charged_energy = charged_energy
        self.discharged_energy = discharged_energy

        # Calculate and print revenues

        self.calculate_and_print_revenues(charged_energy, discharged_energy)

        # Generate plots of the results

        if plot:

            self.plot_results(soc, charged_energy, discharged_energy,alpha, PUN_timeseries[:,1])


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
        soc[0] = soc_0  # Initial state of charge

        c_func = charge_rate_interpolated_func  # Charge rate function
        d_func = discharge_rate_interpolated_func  # Discharge rate function

        for index in range(time_window - 1):

            if c_d_timeseries[index] >= 0:

                # Limit charge based on charge capacity and SoC

                c_d_timeseries[index] = min(c_d_timeseries[index]*alpha[index], min(c_func(soc[index])*alpha[index], 0.9 - soc[index]*alpha[index]))

            else:

                # Limit discharge based on discharge capacity and SoC

                c_d_timeseries[index] = max(c_d_timeseries[index]*alpha[index], max(-d_func(soc[index])*alpha[index], - soc[index]*alpha[index] + 0.1))

            if c_d_timeseries[index] >= 0:

                # Calculate charged energy

                charged_energy[index] = c_d_timeseries[index] * size

            else:

                # Calculate discharged energy

                discharged_energy[index] = c_d_timeseries[index] * size

            # Update SoC for the next time step

            if c_d_timeseries[index] >= 0:

                soc[index + 1] = min(0.9, soc[index] + charged_energy[index]/size)

            else:

                soc[index + 1] = max(0.1, soc[index] + discharged_energy[index]/size)

        return soc, charged_energy, discharged_energy, c_d_timeseries

    def calculate_and_print_revenues(self, charged_energy, discharged_energy):

        """
        Calculates and prints total revenues for the optimized period.

        Args:
            charged_energy (list): Charged energy for each time step.
            discharged_energy (list): Discharged energy for each time step.

        """

        PUN_ts = PUN_timeseries[:,1]  # Time series of energy prices

        # Calculate revenues by summing the costs of charging and discharging

        rev = - (np.array(discharged_energy) * PUN_ts / 1000) - (np.array(charged_energy) * PUN_ts / 1000)

        # Print total revenues

        print("\nRevenues for optimized time window [EUROs]:\n\n", rev.sum())


    def plot_results(self, soc, charged_energy, discharged_energy, alpha, PUN_Timeseries):

        """
        Generates plots of the state of charge, charged energy, discharged energy, and energy prices.

        Args:
            soc (list): State of charge for each time step.
            charged_energy (list): Charged energy for each time step.
            discharged_energy (list): Discharged energy for each time step.

        """
        if plot:

            plots = EnergyPlots(time_window, soc, charged_energy, discharged_energy, PUN_timeseries[:,1])
            plots.plot_soc()
            plots.plot_charged_energy()
            plots.plot_discharged_energy()
            plots.plot_combined_energy_with_pun(num_values=time_window)
            plots.plot_alpha_vs_timewindow(time_window,alpha, PUN_Timeseries)



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
    revenues = -c_d_energy * PUN_timeseries[:,1]
    data = []

    for i in range(len(PUN_timeseries[:,1])):
        entry = {
            "datetime": PUN_timeseries[i, 0].isoformat() + "Z",
            "PUN": PUN_timeseries[i, 1]/1000,
            "soc": SoC[i],
            "c_d_energy": c_d_energy[i]*size,
            "c_d_rate": abs(c_d_energy[i]),
            "revenues": revenues[i],
            "technology": technology,
            "size": size,
            "dod": range_str
            #"source": PUN_timeseries[i, 2]
        }
        data.append(entry)

        json_file_path = output_json_path
        with open(json_file_path, 'w') as json_file:
            json.dump(data, json_file, indent=4)




