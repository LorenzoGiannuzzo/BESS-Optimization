import os
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.colors as mcolors

from matplotlib.colors import Normalize

matplotlib.use('Agg')

class EnergyPlots:
    def __init__(self, time_window, soc, charged_energy, discharged_energy, PUN_timeseries):
        self.time_window = time_window
        self.soc = soc
        self.charged_energy = charged_energy
        self.discharged_energy = discharged_energy
        self.PUN_timeseries = PUN_timeseries
        self.time_steps = np.arange(time_window)
        self.plots_dir = "Plots"
        if not os.path.exists(self.plots_dir):
            os.makedirs(self.plots_dir)

    def plot_soc(self):
        plt.figure(figsize=(12, 8))
        plt.bar(self.time_steps, self.soc, color='grey')
        plt.title('State of Charge (SoC) [%]')
        plt.xlabel('Time Window [h]')
        plt.ylabel('SoC')
        plt.savefig(os.path.join(self.plots_dir, "SoC.png"))
        plt.close()

    def plot_charged_energy(self):
        plt.figure(figsize=(12, 8))
        plt.bar(self.time_steps, self.charged_energy, color='limegreen')
        plt.title('Charged Energy')
        plt.xlabel('Time Window [h]')
        plt.ylabel('Charged Energy [kWh]')
        plt.savefig(os.path.join(self.plots_dir, "Charged_Energy.png"))
        plt.close()

    def plot_discharged_energy(self):
        plt.figure(figsize=(12, 8))
        plt.bar(self.time_steps, self.discharged_energy, color='darkred')
        plt.title('Discharged Energy')
        plt.xlabel('Time Window [h]')
        plt.ylabel('Discharged Energy [kWh]')
        plt.savefig(os.path.join(self.plots_dir, "Discharged_Energy.png"))
        plt.close()

    def plot_combined_energy_with_pun(self, num_values):
        time_steps_24 = self.time_steps[:num_values]
        charged_energy_24 = self.charged_energy[:num_values]
        discharged_energy_24 = self.discharged_energy[:num_values]
        pun_values_24 = self.PUN_timeseries[:num_values]
        soc_24 = self.soc[:num_values]

        fig, (ax0, ax1) = plt.subplots(2, 1, figsize=(12, 12), sharex=True)

        # Normalize SoC values to be in the range [0, 1] for the colormap
        norm = Normalize(vmin=min(soc_24), vmax=max(soc_24))
        cmap = matplotlib.colors.LinearSegmentedColormap.from_list("", ["darkred", "gold", "limegreen"])

        # Plot SoC with gradient colored bars based on value
        for i in range(len(time_steps_24)):
            ax0.bar(time_steps_24[i], soc_24[i], color=cmap(norm(soc_24[i])))

        ax0.set_title('State of Charge (SoC) [%]')
        ax0.set_ylabel('SoC')

        # Plot charged and discharged energy with PUN on the second subplot
        width = 0.4
        ax1.bar(time_steps_24 - width / 2, charged_energy_24, width=width, color='limegreen',
                label='Charged Energy [kWh]')
        ax1.bar(time_steps_24 + width / 2, discharged_energy_24, width=width, color='darkred',
                label='Discharged Energy [kWh]')
        ax1.set_ylabel('Energy [kWh]')
        ax1.set_title('Charged and Discharged Energy with PUN')
        ax1.legend(loc='upper left')

        ax2 = ax1.twinx()
        ax2.plot(time_steps_24, pun_values_24, color='black', label='PUN [Euro/MWh]')
        ax2.set_ylabel('PUN Value')
        ax2.legend(loc='upper right')

        ax1.set_xlabel('Time Window [h]')

        fig.tight_layout()
        plt.savefig(os.path.join(self.plots_dir, "C_D_Energy_with_PUN_and_SOC.png"))
        plt.close()

    @staticmethod
    def c_d_plot(charge_rate, discharge_rate, charge_rate_interpolated_func, discharge_rate_interpolated_func ):
        # Plotting for charge_rate
        plt.figure(figsize=(10, 6))
        plt.plot(charge_rate['SoC [%]'], charge_rate['Charge Rate [kWh/(kWhp*h)]'], label='Charge Rate')
        plt.xlabel('SoC [%]')
        plt.ylabel('Charge Rate [kWh/(kWhp*h)]')
        plt.title('Charge Rate vs SoC')
        plt.legend()
        plt.grid(True)

    # Save the plot as a PNG file
        plt.savefig("Plots/charge_rate_plot.png")
        plt.close()

    # Plotting
        plt.figure(figsize=(10, 6))

    # Plot for charge_rate
        plt.plot(charge_rate['SoC [%]'], charge_rate['Charge Rate [kWh/(kWhp*h)]'], 'o', label='Charge Rate')
        plt.plot(charge_rate['SoC [%]'], charge_rate_interpolated_func(charge_rate['SoC [%]']), '-',
             label='Interpolated Charge Rate')

    # Plot for discharge_rate
        plt.plot(discharge_rate['SoC [%]'], discharge_rate['Discharge Rate [kWh/(kWhp*h)]'], 'o', color='red',
             label='Discharge Rate')
        plt.plot(discharge_rate['SoC [%]'], discharge_rate_interpolated_func(discharge_rate['SoC [%]']), '-',
             color='green', label='Interpolated Discharge rate')

        plt.xlabel('SoC [%]')
        plt.ylabel('Rate [kWh/(kWhp*h)]')
        plt.title('Interpolated Functions')
        plt.legend()
        plt.grid(True)

    # Save the plot as a PNG file
        if not os.path.exists("Plots"):
            os.makedirs("Plots")
        plt.savefig("Plots/interpolated_functions.png")
    # Close the figure to release memory
        plt.close()

    # Plotting for charge_rate
        plt.figure(figsize=(10, 6))
        plt.plot(charge_rate['SoC [%]'], charge_rate['Charge Rate [kWh/(kWhp*h)]'], label='Charge Rate')
        plt.xlabel('SoC [%]')
        plt.ylabel('Charge Rate [kWh/(kWhp*h)]')
        plt.title('Charge Rate vs SoC')
        plt.legend()
        plt.grid(True)

        # Save the plot as a PNG file
        plt.savefig("Plots/charge_rate_plot.png")
        plt.close()

        # Plotting for discharge_rate
        plt.figure(figsize=(10, 6))
        plt.plot(discharge_rate['SoC [%]'], discharge_rate['Discharge Rate [kWh/(kWhp*h)]'], color='red',
                 label='Discharge Rate')
        plt.xlabel('SoC [%]')
        plt.ylabel('Discharge Rate [kWh/(kWhp*h)]')
        plt.title('Discharge Rate vs SoC')
        plt.legend()
        plt.grid(True)

        # Save the plot as a PNG file
        plt.savefig("Plots/discharge_rate_plot.png")
        plt.close()

    @staticmethod
    def total_convergence(n_gen, timewindow, pop_size, X, Y):
        # Convert Y to a DataFrame (if not already)
        Y = pd.DataFrame(Y)

        # Calculate statistics of Y
        Y_stats = Y.transpose().describe()
        Y_stats = Y_stats.transpose()
        # Create a new figure and axis
        fig, ax = plt.subplots(figsize=(10, 6))

        # Define a custom colormap
        cmap_colors = ["orange", "darkorchid", "indigo"]
        cmap = mcolors.LinearSegmentedColormap.from_list("", cmap_colors)

        # Plot the mean from Y_stats with customization
        Y_stats['mean'].plot(ax=ax, color=cmap(0.5), linestyle='-', linewidth=2,
                             label='Mean Fitness Value of Population')  # Customize the plot with linestyle, linewidth, and label
        ax.legend()

        # Add titles and axis labels
        ax.set_title('Statistics of Fitness', fontsize=16)  # Title of the plot
        ax.set_xlabel('Categories', fontsize=14)  # X-axis label
        ax.set_ylabel('Values', fontsize=14)  # Y-axis label

        # Customize background and grid
        ax.set_facecolor('whitesmoke')  # Background color
        ax.grid(True, linestyle='--', linewidth=0.5,
                color='white')  # Dashed grid lines with 0.5 linewidth and gray color

        # Save the figure
        plt.savefig('Plots/total_convergence.png')

    @staticmethod
    def PUN_plot(PUN_timeseries):
        # Line plot dei valori PUN (terza colonna del DataFrame)
        pun_values = PUN_timeseries  # Estrazione della terza colonna (indice 2)
        plt.figure(figsize=(12, 8))
        plt.plot(pun_values, marker='o', color='b')
        plt.title('PUN Values')
        plt.xlabel('Time step')
        plt.ylabel('PUN Value')
        plt.savefig(os.path.join("Plots", "PUN_values_plot.png"))
        plt.close()

    @staticmethod
    def convergence(n_gen, timewindow, pop_size, X, Y, max_subplots_per_figure=72):
        # Define the timesteps
        timesteps = np.arange(0, n_gen)

        # Create a colormap based on a range of darker colors
        cmap = matplotlib.colors.LinearSegmentedColormap.from_list("", ["orange", "darkorchid", "indigo"])

        # Normalize Y using a logarithmic scale to improve color distribution
        norm = Normalize(vmin=np.min(Y), vmax=np.max(Y))

        # Number of figures needed
        num_figures = (timewindow + max_subplots_per_figure - 1) // max_subplots_per_figure

        for fig_num in range(num_figures):
            # Determine the number of subplots in this figure
            start = fig_num * max_subplots_per_figure
            end = min(start + max_subplots_per_figure, timewindow)
            num_subplots = end - start

            # Determine the grid size for the subplots
            rows = int(np.ceil(np.sqrt(num_subplots)))
            cols = int(np.ceil(num_subplots / rows))

            # Create a figure and a grid of subplots
            fig, axes = plt.subplots(rows, cols, figsize=(20, 18))

            # Flatten the array of axes to iterate over them
            axes = axes.flatten()

            # Iterate over each subplot in the current figure
            for k in range(num_subplots):
                ax = axes[k]

                # Prepare data for the current subplot
                for i in range(pop_size):
                    # Calculate colors based on normalized Y values for individual i
                    colors = cmap(norm(Y[:, i])*10)
                    ax.scatter(timesteps, X[:, i, start + k], s=10, alpha=0.8, c=colors)

                # Set title and labels for the subplot
                ax.set_title(f'C/D Energy % at {start + k + 1}h')
                ax.set_xlabel('Generations')
                ax.set_ylabel('% of C/D')
                ax.grid(True)

            # Hide any empty subplots if present
            for k in range(num_subplots, len(axes)):
                fig.delaxes(axes[k])

            # Add spacing between subplots
            plt.tight_layout()

            # Check if the "Plots" folder exists, create it if not
            output_dir = 'Plots'
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)

            # Save the figure in the "Plots" folder
            output_path = os.path.join(output_dir, f'convergence_{fig_num + 1}.png')
            fig.savefig(output_path)
            plt.close(fig)



