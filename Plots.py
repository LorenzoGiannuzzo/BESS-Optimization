import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
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
        cmap = matplotlib.colors.LinearSegmentedColormap.from_list("", ["darkred","gold","limegreen"])

        # Plot SoC with gradient colored bars based on value
        for i in range(len(time_steps_24)):
            ax0.bar(time_steps_24[i], soc_24[i], color=cmap(norm(soc_24[i])))

        ax0.set_title('State of Charge (SoC) [%]')
        ax0.set_ylabel('SoC')

        # Plot charged and discharged energy with PUN on the second subplot
        width = 0.4
        ax1.bar(time_steps_24 - width / 2, charged_energy_24, width=width, color='limegreen', label='Charged Energy [kWh]')
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

def convergence(n_gen, timewindow, pop_size, X, Y):

    # Define the timesteps
    timesteps = np.arange(0, n_gen)

    # Create a figure and a grid of subplots
    fig, axes = plt.subplots(9, 8, figsize=(20, 18))

    # Flatten the array of axes to iterate over them
    axes = axes.flatten()

    # Create a colormap based on a range of darker colors
    cmap = matplotlib.colors.LinearSegmentedColormap.from_list("", ["orange", "darkorchid", "indigo"])

    # Normalize Y using a logarithmic scale to improve color distribution
    norm = Normalize(vmin=np.min(Y), vmax=np.max(Y))

    # Iterate over each subplot
    for k in range(timewindow):
        ax = axes[k]

        # Prepare data for the current subplot
        for i in range(pop_size):
            # Calculate colors based on normalized Y values for individual i,
            # scaled by a factor of 10000
            colors = cmap(norm(Y[:, i]) * 10000)
            ax.scatter(timesteps, X[:, i, k], s=10, alpha=0.8, c=colors)

        # Set title and labels for the subplot
        ax.set_title(f'C/D Energy % at {k + 1}h')
        ax.set_xlabel('Generations')
        ax.set_ylabel('% of C/D')
        ax.grid(True)

    # Hide any empty subplots if present
    for k in range(timewindow, len(axes)):
        fig.delaxes(axes[k])

    # Remove the colorbar (commented out section)
    # sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    # sm.set_array([])
    # cbar = fig.colorbar(sm, ax=axes.tolist(), orientation='vertical', fraction=0.02, pad=0.01)
    # cbar.set_label('Normalized Y values')

    # Add spacing between subplots
    plt.tight_layout()

    # Check if the "Plots" folder exists, create it if not
    output_dir = 'Plots'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Save the figure in the "Plots" folder
    output_path = os.path.join(output_dir, 'convergence.png')
    fig.savefig(output_path)