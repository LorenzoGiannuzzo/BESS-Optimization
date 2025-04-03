"""

BESS Optimization using NSGA-III Algorithm

    __author__ = "Lorenzo Giannuzzo"
    __maintainer__ = "Lorenzo Giannuzzo"
    __email__ = "lorenzo.giannuzzo@polito.it"
    __status__ = "in progress"
    __version__ = "v0.2.1"
    __license__ = "MIT"

Last Update of current code: 01/04/2025 - 17:49

"""

# IMPORT LIBRARIES
import os
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.colors as mcolors
import matplotlib.gridspec as gridspec
from matplotlib.colors import Normalize
from argparser_l import size, POD_power

matplotlib.use('Agg')


# DEFINE ENERGY PLOT CLASS
class EnergyPlots:

    def __init__(self, time_window, soc, charged_energy, discharged_energy, PUN_timeseries, taken_from_grid,
                 taken_from_pv, produced_from_pv,discharged_from_pv,self_consumption,from_pv_to_load,
                 from_BESS_to_laod,load):

        self.time_window = time_window

        self.soc = soc
        self.charged_energy = np.array(charged_energy)
        self.discharged_energy = np.array(discharged_energy)
        self.PUN_timeseries = np.array(PUN_timeseries)
        self.time_steps = np.array(np.arange(time_window))
        self.taken_from_grid = np.array(taken_from_grid)
        self.taken_from_pv = np.array(taken_from_pv)
        self.produced_from_pv = np.array(produced_from_pv)
        self.discharged_from_pv = np.array(discharged_from_pv)
        self.plots_dir = "Plots/Results/Long Simulation"
        self.self_consumption = self_consumption
        self.from_pv_to_load = np.array(from_pv_to_load)
        self.from_BESS_to_load = np.array(from_BESS_to_laod)
        self.load = load

        if not os.path.exists(self.plots_dir):
            os.makedirs(self.plots_dir)

    # DEFINE SoC PLOT FUNCTION
    def plot_soc(self):

        # SET FIG SIZE
        plt.figure(figsize=(12, 8))
        plt.bar(self.time_steps, self.soc, color='grey')
        plt.title('State of Charge (SoC) [%]')
        plt.xlabel('Time Window [h]')
        plt.ylabel('SoC')
        plt.savefig(os.path.join(self.plots_dir, "SoC.png"))

    # DEFINE CHARGED ENERGY PLOT FUNCTION
    def plot_charged_energy(self):

        # SET FIG SIZE
        plt.figure(figsize=(12, 8))
        plt.bar(self.time_steps, self.charged_energy, color='limegreen')
        plt.title('Charged Energy')
        plt.xlabel('Time Window [h]')
        plt.ylabel('Charged Energy [kWh]')
        plt.savefig(os.path.join(self.plots_dir, "Charged_Energy.png"))

    # DEFINE DISCHARGED ENERGY FUNCTION
    def plot_discharged_energy(self):

        plt.figure(figsize=(12, 8))
        plt.bar(self.time_steps, self.discharged_energy, color='darkred')
        plt.title('Discharged Energy')
        plt.xlabel('Time Window [h]')
        plt.ylabel('Discharged Energy [kWh]')
        plt.savefig(os.path.join(self.plots_dir, "Disc_Energy.png"))

    import numpy as np
    import matplotlib.pyplot as plt
    import matplotlib
    from matplotlib.colors import Normalize
    import os

    def Total_View(self, num_values):
        # GET VARIABLES FROM SELF
        time_steps = self.time_steps[:num_values]
        discharged_energy = self.discharged_energy[:num_values]
        pun_values = self.PUN_timeseries[:num_values]
        taken_from_pv = self.taken_from_pv[:num_values]  # Energy taken from PV
        taken_from_grid = self.taken_from_grid[:num_values]  # Energy taken from the grid
        pv_prodcution = self.produced_from_pv[:num_values]
        from_pv_to_load = self.from_pv_to_load[:num_values]
        from_bess_to_load = self.from_BESS_to_load[:num_values]
        from_pv_to_grid = self.discharged_from_pv[:num_values]
        soc = self.soc[:num_values]  # Assuming you have a soc attribute in self
        load = self.load[:num_values]

        # EVALUATE REVENUES
        rev = (- (np.array(discharged_energy) * pun_values / 1000) - (
                taken_from_grid * pun_values / 1000) + (
                   -np.array(discharged_energy)) * pun_values / 1000)  # Adjusted for clarity
        rev = np.array(rev, dtype=float)

        # PLOTTING DISCHARGED ENERGY, TAKEN FROM PV, AND TAKEN FROM GRID
        bar_width = 0.35  # Adjusted bar width
        index = np.arange(num_values)

        plt.figure(figsize=(14, 14))  # Increased figure size for better visibility

        # Create primary axis for energy flows
        ax1 = plt.subplot(211)  # First subplot for energy flows

        # Bar plot for discharged energy
        bar1 = ax1.bar(index, discharged_energy, bar_width, label='BESS to Grid', color='darkred')

        # Fill areas for produced from PV and electrical load
        fill1 = ax1.fill_between(index, 0, pv_prodcution, color='red', alpha=0.1,
                                 label='Produced from PV')

        load = np.array(load, dtype=float)

        fill2 = ax1.fill_between(index, 0, load, color='lightblue', alpha=0.3,
                                 label='Electrical Load')

        # Bar plot for taken from PV (lighter green)
        bar3 = ax1.bar(index, taken_from_pv, bar_width, label='PV to BESS', color='lightgreen', alpha=0.8,
                       bottom=from_pv_to_load)

        # Bar plot for taken from grid (darker green) - now paired with other bars
        bar2 = ax1.bar(index + bar_width, taken_from_grid, bar_width, label='Grid to BESS', color='darkgreen',
                       alpha=0.8)

        # Bar plot for from PV to load
        bar4 = ax1.bar(index, from_pv_to_load, bar_width, label='PV to Load', color='indianred', alpha=0.8)

        # Bar plot for from BESS to load, starting from the top of the discharged energy bar
        bar5 = ax1.bar(index, from_bess_to_load, bar_width, label='BESS to Load', color='steelblue', alpha=0.8,
                       bottom=from_pv_to_load)

        bar6 = ax1.bar(index, -from_pv_to_grid, bar_width, label='PV to Grid', color='aquamarine', alpha=0.8,
                       bottom=taken_from_pv + from_pv_to_load)

        # Create a second y-axis for pun_values
        ax2 = ax1.twinx()
        line, = ax2.plot(index + bar_width / 2, pun_values, label='PUN Values', color='black', linewidth=2)

        # Set labels and title
        ax1.set_xlabel('Hours', fontsize=16)  # Increased font size for better visibility
        ax1.set_ylabel('Energy [kWh]', fontsize=16)  # Increased font size for better visibility
        ax2.set_ylabel('Zonal Price [Euros/kWh]', fontsize=16)  # Adjust the label as needed
        plt.title('Energy Flows', fontsize=16)  # Increased font size for better visibility
        ax1.set_xticks(index + bar_width / 2)
        ax1.set_xticklabels(time_steps, rotation=0)

        # Create separate legends
        # Legend for fill areas
        fill_handles = [fill1, fill2]
        fill_labels = ['PV Production', 'Electrical Load']
        ax1.legend(fill_handles, fill_labels, loc='upper right', fontsize=16)  # Place in upper left

        # Legend for bar plots
        bar_handles = [bar1, bar2, bar3, bar4, bar5, bar6]
        bar_labels = ['BESS to Grid', 'Grid to BESS', 'PV to BESS', 'PV to Load',
                      'BESS to Load', 'PV to Grid']
        ax2.legend(bar_handles, bar_labels, loc='upper left', fontsize=16)  # Place in upper right

        max_energy = np.max(taken_from_grid + taken_from_pv + from_pv_to_load + from_bess_to_load + discharged_energy)
        min_energy = np.min(np.array(discharged_energy) - np.abs(from_bess_to_load), 0)
        ax1.set_ylim(min_energy * 1.1, max_energy * 1.1)

        # Add horizontal dashed gray grid lines
        ax1.yaxis.grid(True, linestyle='--', color='gray', alpha=0.3)  # Adjust alpha for visibility

        # Create a second subplot for SoC
        ax0 = plt.subplot(212)  # Second subplot for SoC

        # Normalize SoC for color mapping
        norm = Normalize(vmin=min(soc), vmax=max(soc))
        cmap = matplotlib.colors.LinearSegmentedColormap.from_list("", ["lightblue", "steelblue", "darkblue"])

        # Plot SoC
        ax0.bar(index, np.array(soc) * float(100), color=cmap(norm(soc)), width=bar_width)
        ax0.set_title('State of Charge (SoC)', fontsize=16)
        ax0.set_ylabel('SoC [%]', fontsize=16)
        ax0.set_xlabel('Hours', fontsize=16)  # Label for x-axis
        ax0.set_ylim(0, 100)  # Assuming SoC is a percentage
        ax0.grid(axis='y', linestyle='--', alpha=0.3)

        # Adjust layout to make room for titles and labels
        plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1)
        plt.tight_layout()

        # Save the plot
        plt.savefig(os.path.join(self.plots_dir, "Total_View.pdf"), dpi=500)
        plt.close()

    @staticmethod
    def plot_degradation():

        from BESS_model_l import degradation

        # Generate cycle numbers from 0 to 7000
        cycles = np.arange(0, 7001,1,dtype=float)  # 7001 to include 7000
        capacities = degradation(cycles)  # Calculate capacities for each cycle

        # Create the plot
        plt.figure(figsize=(10, 6))
        plt.plot(np.arange(0,7001,1,dtype=float), degradation(np.arange(0, 7001,1,dtype=float)), label='Capacity Remaining')
        plt.xlabel('Number of Cycles')
        plt.ylabel('Capacity Remaining (%)')
        plt.title('Battery Capacity Degradation Over Cycles')
        plt.grid(True)
        plt.legend()
        plt.savefig(os.path.join('Plots/General', 'Degradation.png'), dpi=300, bbox_inches='tight')

    @staticmethod
    def c_d_plot(charge_rate, discharge_rate):
        # Plotting for charge_rate
        plt.figure(figsize=(10, 6))
        plt.plot(charge_rate['SoC [%]'], charge_rate['Charge Rate [kWh/(kWhp*h)]'], label='Charge Rate')
        plt.xlabel('SoC [%]', fontsize=16)  # Increased font size for x-axis label
        plt.ylabel('Charge Rate [kWh/(kWhp*h)]', fontsize=16)  # Increased font size for y-axis label
        plt.title('Charge Rate vs SoC', fontsize=18)  # Increased font size for title
        plt.legend(fontsize=14)
        plt.grid(True)

        # Save the plot as a PNG file with better quality
        plt.savefig("Plots/General/charge_rate.png", dpi=500)  # Increased DPI for better quality
        plt.close()

        # Plotting for charge_rate and discharge_rate
        plt.figure(figsize=(10, 6))

        # Plot for charge_rate
        plt.plot(charge_rate['SoC [%]'], charge_rate['Charge Rate [kWh/(kWhp*h)]'], 'o', label='Charge Rate',markersize=10)

        # Plot for discharge_rate
        plt.plot(discharge_rate['SoC [%]'], discharge_rate['Discharge Rate [kWh/(kWhp*h)]'], 'o', color='red',
                 label='Discharge Rate', markersize=10)

        plt.xlabel('SoC [%]', fontsize=16)  # Increased font size for x-axis label
        plt.ylabel('Rate [%]', fontsize=16)  # Increased font size for y-axis label
        plt.title('Interpolated Functions', fontsize=18)  # Increased font size for title
        plt.legend(fontsize=14)
        plt.grid(True)

        # Save the plot as a PNG file with better quality
        plt.savefig("Plots/General/interpolated_functions.png", dpi=500)  # Increased DPI for better quality
        plt.close()

        # Plotting for charge_rate
        plt.figure(figsize=(10, 6))
        plt.plot(charge_rate['SoC [%]'] * 100, charge_rate['Charge Rate [kWh/(kWhp*h)]'] * 100, label='Charge Rate')
        plt.xlabel('SoC [%]', fontsize=20)  # Increased font size for x-axis label
        plt.ylabel('Charge Rate [%]', fontsize=20)  # Increased font size for y-axis label
        plt.title('Charge Rate vs SoC', fontsize=20)  # Increased font size for title
        plt.xticks(fontsize=16)
        plt.yticks(fontsize=16)
        plt.tight_layout()  # Adjust layout to avoid overlap
        plt.grid(True)

        # Save the plot as a TIFF file with high resolution
        plt.savefig("Plots/General/charge_rate.tiff", dpi=500)  # Increased DPI for better quality
        plt.close()

        # Plotting for discharge_rate
        plt.figure(figsize=(10, 6))
        plt.plot(discharge_rate['SoC [%]'] * 100, discharge_rate['Discharge Rate [kWh/(kWhp*h)]'] * 100, color='red',
                 label='Discharge Rate')
        plt.xlabel('SoC [%]', fontsize=20)  # Increased font size for x-axis label
        plt.ylabel('Discharge Rate [%]', fontsize=20)  # Increased font size for y-axis label
        plt.title('Discharge Rate vs SoC', fontsize=20)  # Increased font size for title
        plt.xticks(fontsize=16)
        plt.yticks(fontsize=16)
        plt.tight_layout()  # Adjust layout to avoid overlap
        plt.grid(True)

        # Save the plot as a TIFF file with high resolution
        plt.savefig("Plots/General/disc_rate.tiff", dpi=500)


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
        ax.set_xlabel('Generations', fontsize=14)  # X-axis label
        ax.set_ylabel('OF Mean Value', fontsize=14)  # Y-axis label

        # Customize background and grid
        ax.set_facecolor('whitesmoke')  # Background color
        ax.grid(True, linestyle='--', linewidth=0.5,
                color='white')  # Dashed grid lines with 0.5 linewidth and gray color

        # Save the figure
        plt.savefig('Plots/Results/Long Simulation/total_convergence.png')

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
                    ax.scatter(timesteps, X[:, i, start + k], s=4, alpha=0.8, c=colors)

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
            output_dir = 'Plots/Results/Long Simulation'

            if not os.path.exists(output_dir):
                os.makedirs(output_dir)

            # Save the figure in the "Plots" folder
            output_path = os.path.join(output_dir, f'convergence_{fig_num + 1}.png')
            fig.savefig(output_path)

    def plot_daily_energy_flows(self, num_values):

        time_steps = self.time_steps[:num_values]
        charged_energy = self.charged_energy[:num_values]
        discharged_energy = self.discharged_energy[:num_values]
        pun_values = self.PUN_timeseries[:num_values]
        soc = self.soc[:num_values]
        produced_from_pv = self.produced_from_pv[:num_values]
        taken_from_pv = self.taken_from_pv[:num_values]
        taken_from_grid = self.taken_from_grid[:num_values]
        discharged_from_pv = self.discharged_from_pv[:num_values]

        rev = - (np.array(discharged_energy) * pun_values / 1000) - (taken_from_grid * pun_values / 1000) - (
                discharged_from_pv * pun_values / 1000)
        rev_cumulative = np.cumsum(rev)

        num_settimane = 12
        ore_per_settimana = 24
        weeks = [slice(i * ore_per_settimana, (i + 1) * ore_per_settimana) for i in range(num_settimane)]
        month_names = [
            "January", "February", "March", "April", "May", "June",
            "July", "August", "September", "October", "November", "December"
        ]

        fig = plt.figure(figsize=(16, 48))
        gs = gridspec.GridSpec(num_settimane, 2)
        width = 0.6

        for i, week in enumerate(weeks):
            # Asse per SoC
            ax0 = fig.add_subplot(gs[i, 0])  # Posizione del box per SoC
            norm = Normalize(vmin=min(soc), vmax=max(soc))
            cmap = matplotlib.colors.LinearSegmentedColormap.from_list("", ["lightblue", "steelblue", "darkblue"])

            # Plot SoC
            ax0.bar(time_steps[week], soc[week], color=cmap(norm(soc[week])), width=width)
            ax0.set_title(f'Typical Day of {month_names[i]} - State of Charge', fontsize=14)
            ax0.set_ylabel('SoC [%]', fontsize=12)
            # Set limits for SoC
            ax0.grid(axis='y', linestyle='--', alpha=0.4)

            # Energy axis
            ax1 = fig.add_subplot(gs[i, 1])
            ax1.set_title(f'Typical Day of {month_names[i]} - Energy Flows', fontsize=14)
            ax1.fill_between(time_steps[week], 0, produced_from_pv[week], color='lightblue', alpha=0.3,
                             label='Produced from PV')
            ax1.bar(time_steps[week], taken_from_grid[week], color='darkgreen', label='From Grid to BESS', width=0.4)
            ax1.bar(time_steps[week], taken_from_pv[week], color='darkblue', bottom=-discharged_from_pv[week],
                    width=0.4, label='From PV to BESS')
            ax1.bar(time_steps[week], discharged_energy[week], color='darkred', label='From BESS to Grid', width=0.4)
            ax1.bar(time_steps[week], -discharged_from_pv[week], color='steelblue', label='From PV to Grid',
                    width=0.4)

            ax1.set_ylabel('Energy [ kWh]', fontsize=12)
            ax1.legend(loc='upper left', fontsize=10)
            ax1.grid(axis='y', linestyle='--', alpha=0.7)

            # Plot PUN values on the secondary axis
            ax2 = ax1.twinx()
            ax2.plot(time_steps[week], pun_values[week], color='black', label='PUN', linewidth=2)
            ax2.set_ylabel('PUN [Euro/MWh]', fontsize=12)
            ax2.legend(loc='upper right', fontsize=10)

        plt.subplots_adjust(hspace=0.4, wspace=0.3)  # Aumenta lo spazio verticale e orizzontale tra i grafici
        plt.tight_layout()
        plt.savefig(os.path.join(self.plots_dir, "Daily_Energy_Flows.png"))










