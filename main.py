import numpy as np

from objective_function import Revenues, pop_size, time_window, soc_0
from BESS_parameters import charge_rate_interpolated_func, discharge_rate_interpolated_func, size, charge_rate, discharge_rate
from Economic_parameters import PUN_timeseries
from Optimizer import Optimizer
from Plots import EnergyPlots


class Main:
    def __init__(self):
        """
        Initializes the Main object. Creates an instance of the objective function (Revenues) and the optimizer
        (Optimizer).
        """
        self.objective_function = Revenues()
        self.optimizer = Optimizer(objective_function=self.objective_function, pop_size=pop_size)

    def run_optimization(self):
        """
        Executes the optimization, applies physical constraints, calculates revenues, and generates plots.
        """
        # Execute the optimization and get the solution
        solution = self.optimizer.maximize_revenues()
        self.history = solution.history
        # self.history[i].pop[p].get(a) dove i= iterazione, p= individuo della popolazione, a=attributo tipi X o f
        # Get the charge/discharge time series from the solution
        c_d_timeseries = solution.X

        # Apply physical constraints to the charge/discharge time series
        soc, charged_energy, discharged_energy = self.apply_physical_constraints(c_d_timeseries)

        # Calculate and print revenues
        self.calculate_and_print_revenues(charged_energy, discharged_energy)

        # Generate plots of the results
        self.plot_results(soc, charged_energy, discharged_energy)

    def apply_physical_constraints(self, c_d_timeseries):
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
                c_d_timeseries[index] = min(c_d_timeseries[index], min(c_func(soc[index]), 0.9 - soc[index]))
            else:
                # Limit discharge based on discharge capacity and SoC
                c_d_timeseries[index] = max(c_d_timeseries[index], max(-d_func(soc[index]), -soc[index] + 0.1))

            if c_d_timeseries[index] >= 0:
                # Calculate charged energy
                charged_energy[index] = c_d_timeseries[index] * size
            else:
                # Calculate discharged energy
                discharged_energy[index] = c_d_timeseries[index] * size

            # Update SoC for the next time step
            if c_d_timeseries[index] >= 0:
                soc[index + 1] = min(0.9, soc[index] + charged_energy[index] / size)
            else:
                soc[index + 1] = max(0.1, soc[index] + discharged_energy[index] / size)

        return soc, charged_energy, discharged_energy

    def calculate_and_print_revenues(self, charged_energy, discharged_energy):
        """
        Calculates and prints total revenues for the optimized period.

        Args:
            charged_energy (list): Charged energy for each time step.
            discharged_energy (list): Discharged energy for each time step.
        """
        PUN_ts = PUN_timeseries  # Time series of energy prices
        # Calculate revenues by summing the costs of charging and discharging
        rev = - (np.array(discharged_energy) * PUN_ts / 1000) - (np.array(charged_energy) * PUN_ts / 1000)
        # Print total revenues
        print("\nRevenues for optimized time window [EUROs]:\n\n", rev.sum())

    def plot_results(self, soc, charged_energy, discharged_energy):
        """
        Generates plots of the state of charge, charged energy, discharged energy, and energy prices.

        Args:
            soc (list): State of charge for each time step.
            charged_energy (list): Charged energy for each time step.
            discharged_energy (list): Discharged energy for each time step.
        """
        plots = EnergyPlots(time_window, soc, charged_energy, discharged_energy, PUN_timeseries)
        plots.plot_soc()
        plots.plot_charged_energy()
        plots.plot_discharged_energy()
        plots.plot_combined_energy_with_pun(num_values=time_window)


if __name__ == "__main__":
    # Create an instance of the Main class
    main = Main()
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

    EnergyPlots.PUN_plot(PUN_timeseries)
    EnergyPlots.convergence(len(main.history),time_window, pop_size, X, Y)
    EnergyPlots.c_d_plot(charge_rate, discharge_rate, charge_rate_interpolated_func, discharge_rate_interpolated_func)
    EnergyPlots.total_convergence(len(main.history), time_window, pop_size, X, Y)



