""" BESS Optimization using NSGA-III Algorithm

    __author__ = "Lorenzo Giannuzzo"
    __maintainer__ = "Lorenzo Giannuzzo"
    __email__ = "lorenzo.giannuzzo@polito.it"
    __status__ = "in progress"
    __version__ = "v0.2.1"
    __license__ = "MIT"

Last Update of current code: 05/06/2025 """

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
                 from_BESS_to_laod, shared_energy_bess, load, rec_load, rec_production):

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
        self.shared_energy_bess = shared_energy_bess
        self.rec_load = rec_load
        self.rec_production = rec_production

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

    # DEFINE COMBINED PLOTS FIGURE FUCNTION
    def plot_combined_energy_with_pun(self, num_values):

        # SETTING VALUES
        time_steps_24 = self.time_steps[:num_values]
        charged_energy_24 = self.charged_energy[:num_values]
        discharged_energy_24 = self.discharged_energy[:num_values]
        pun_values_24 = self.PUN_timeseries[:num_values]
        soc_24 = self.soc[:num_values]
        produced_from_pv = self.produced_from_pv[:num_values]
        taken_from_pv_24 = self.taken_from_pv[:num_values]
        taken_from_grid_24 = self.taken_from_grid[:num_values]
        discharged_from_pv = self.discharged_from_pv

        # EVALUATE PV AND BESS REVENUES
        rev_bess = -(np.array(discharged_energy_24) * pun_values_24 / 1000) - (
                taken_from_grid_24 * pun_values_24 / 1000)

        rev_bess = np.array(rev_bess, dtype=float)

        # EVALUATE CUMULATIVE GLOBAL REVENUES
        rev_cumulative = np.cumsum(rev_bess)

        # EVALUATE TOTAL ENERGY TAKEN FROM PV
        total_from_pv = np.sum(taken_from_pv_24)

        # EVALUATE TOTAL ENERGY TAKEN FROM GRID
        total_from_grid = np.sum(taken_from_grid_24)

        # EVALUATE TOTAL ENERGY DISCHARGED FROM BESS
        total_discharged = np.sum(-discharged_energy_24)

        # CREATING FIGURE LAYOUT
        fig = plt.figure(figsize=(24, 12))
        gs = gridspec.GridSpec(2, 2, width_ratios=[1, 1], height_ratios=[1, 1])

        # SET AXIS FOR SoC PLOT (UPPER LEFT)
        ax0 = fig.add_subplot(gs[0, 0])

        # Normalize SoC values to be in the range [0, 1] for the colormap
        norm = Normalize(vmin=min(soc_24*100), vmax=max(soc_24*100))
        cmap = matplotlib.colors.LinearSegmentedColormap.from_list("", ["lightblue", "steelblue", "darkblue"])

        # Plot SoC with gradient colored bars based on value
        for i in range(len(time_steps_24)):
            ax0.bar(time_steps_24[i], soc_24[i] * 100, color=cmap(norm(soc_24[i])))

        ax0.set_title('State of Charge (SoC)')
        ax0.set_ylabel('SoC [%]')

        # SET AXIS FOR BOTTOM LEFT GRAPH
        ax1 = fig.add_subplot(gs[1, 0])
        width = 0.4
        ax1.bar(time_steps_24, [1] * np.array(taken_from_grid_24), width=width, color='darkgreen',
                label='From Grid to BESS')
        ax1.bar(time_steps_24, [1] * np.array(taken_from_pv_24), width=width, color='darkblue',
                label='From PV to BESS')
        ax1.bar(time_steps_24, discharged_energy_24, width=width, color='darkred',
                bottom= np.array(taken_from_grid_24),
                label='From BESS to Grid')

        ax1.set_ylabel('Energy [kWh]')
        ax1.set_title('BESS Energy Flows')
        ax1.legend(loc='upper left')
        plt.ylim(-size * 0.6, size * 0.6)

        # Plot PUN values on the secondary axis
        ax3 = ax1.twinx()
        ax3.plot(time_steps_24, pun_values_24, color='black', label='PUN')
        ax3.set_ylabel('PUN [Euro/MWh]')
        ax3.legend(loc='upper right')
        ax1.set_xlabel('Time Window [h]')

        ax2 = fig.add_subplot(gs[0, 1])  # Usa solo la prima riga nella colonna di destra
        ax2.plot(time_steps_24, rev_cumulative, color='lightgreen', label='Cumulative Revenues', alpha=1)
        ax2.set_title('Cumulative Revenues Over Time')
        ax2.fill_between(time_steps_24, rev_cumulative, color='green', alpha=0.3)  # Area sottesa con alpha 0.3
        ax2.set_ylabel('Cumulative BESS Revenues [Euros]')
        ax2.legend(loc='upper left')

        def func(pct, allvalues):
            absolute = int(np.round(pct / 100. * np.sum(allvalues)))  # Calculate absolute value
            return f"{absolute} ({pct:.1f}%)"  # Format as "absolute (percentage)"

        # Create the pie chart
        ax4 = fig.add_subplot(gs[1, 1])

        sizes = [total_from_pv, total_from_grid, total_discharged]
        labels = ['From PV to BESS', 'From Grid to BESS', 'From BESS to Grid']
        colors = ['Steelblue', 'lightblue', 'paleturquoise']
        explode = (0.05, 0.05, 0.05)  # explode the first slice (Total Curtailment)

        # Create pie chart with custom autopct function
        ax4.pie(sizes, explode=explode, labels=labels, colors=colors,
                autopct=lambda pct: func(pct, sizes),  # Use the custom function
                shadow=False, startangle=90, textprops={'fontsize': 14})

        ax4.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
        ax4.set_title('BESS Energy Distribution [kWh]')
        fig.tight_layout()
        plt.savefig(os.path.join(self.plots_dir, "BESS_View.png"))

    # CREATING PV_View
    def PV_View(self, num_values):

        # INITIALIZE ENERGY VECTORS AND TIME WINDOW
        time_steps = self.time_steps[:num_values]
        charged_energy = self.charged_energy[:num_values]
        discharged_energy = self.discharged_energy[:num_values]
        pun_values = self.PUN_timeseries[:num_values]
        soc = self.soc[:num_values]
        produced_from_pv = self.produced_from_pv[:num_values]
        taken_from_pv = self.taken_from_pv[:num_values]
        taken_from_grid = self.taken_from_grid[:num_values]
        discharged_from_pv = self.discharged_from_pv[:num_values]

        # EVALUATING TOTAL CURTAILMENT
        curtailment = np.maximum(produced_from_pv + discharged_from_pv - taken_from_pv, 0.0)
        total_curtailment = np.sum(curtailment)

        # EVALUATING TOTAL ENERGY TAKEN FROM PV
        total_taken_from_pv = np.sum(taken_from_pv)

        # EVALUATING TOTAL ENERGY DISCHARGED FROM PV
        total_discharged_from_pv = np.sum(-discharged_from_pv)

        # EVALAUTE PV REVENUES
        rev_pv = (discharged_from_pv) * pun_values / 1000
        rev_pv = np.array(rev_pv,dtype=float)

        # EVALUATE PV CUMULATIVE REVENUES
        rev_cumulative = np.cumsum(rev_pv)

        # layout wtih 4 box using gridspec
        fig = plt.figure(figsize=(24, 12))
        gs = gridspec.GridSpec(2, 2, width_ratios=[1, 1], height_ratios=[1, 1])

        # Axis for SoC palce up left
        ax0 = fig.add_subplot(gs[0, 0])

        # Normalize SoC values to be in the range [0, 1] for the colormap
        norm = Normalize(vmin=min(soc*100), vmax=max(soc*100))
        cmap = matplotlib.colors.LinearSegmentedColormap.from_list("", ["lightblue", "steelblue", "darkblue"])

        # Plot SoC with gradient colored bars based on value
        for i in range(len(time_steps)):
            if taken_from_pv[max(i-1,0.0)] == 0:
                color = 'lightgray'  # Set color to gray if taken_from_pv is zero
            else:
                color = cmap(norm(soc[i]))  # Use colormap for the current timestep

            # Plot the current SoC bar
            ax0.bar(time_steps[i], soc[i]*100, color=color)

        ax0.set_title('State of Charge - BESS')
        ax0.set_ylabel('SoC [%]')

        # Axis for charged and discharged energy (second graph) bottom left
        ax1 = fig.add_subplot(gs[1, 0])
        ax1.fill_between(time_steps, 0, produced_from_pv, color='lightblue', alpha=0.3, label='Produced from PV')
        width = 0.4

        ax1.bar(time_steps, taken_from_pv, color='darkblue', bottom=-discharged_from_pv,
                label='From PV to BESS', width=width)
        ax1.bar(time_steps, -discharged_from_pv,
                label='From PV to Grid', width=width)

        ax1.set_ylabel('Energy [kWh]')
        ax1.set_title('PV Energy Flows')
        ax1.legend(loc='upper left')
        ax1.set_xlabel('Time Window [h]')
        plt.ylim(-size * 0.6, size * 0.6)

        # Plot PUN values on the secondary axis
        ax3 = ax1.twinx()
        ax3.plot(time_steps, pun_values, color='black', label='PUN')
        ax3.set_ylabel('PUN [Euro/MWh]')
        ax3.legend(loc='upper right')

        # Cumulative revenue axis to the right of the first graph
        ax2 = fig.add_subplot(gs[0, 1])
        ax2.plot(time_steps, -rev_cumulative, color='lightgreen', label='Cumulative Revenues', alpha=1)
        ax2.set_title('Cumulative Revenues Over Time')
        ax2.fill_between(time_steps, -rev_cumulative, color='green', alpha=0.3)  # Area sottesa con alpha 0.3
        ax2.set_ylabel('Cumulative PV Revenues [Euros]')
        ax2.legend(loc='upper left')

        # New pie chart in the bottom right-hand corner
        ax4 = fig.add_subplot(gs[1, 1])

        if total_curtailment != 0.0:
            def func(pct, allvalues):
                absolute = int(np.round(pct / 100. * np.sum(allvalues)))  # Calculate absolute value
                return f"{absolute} ({pct:.1f}%)"  # Format as "absolute (percentage)"

            # Values for the pie chart
            # Create the pie chart
            sizes = [total_curtailment, total_taken_from_pv, total_discharged_from_pv]
            labels = ['Curtailment', 'From PV to BESS', 'From PV to Grid']
            colors = ['orange', 'steelblue', 'khaki']
            explode = (0.05, 0.05, 0.05)  # explode the first slice (Total Curtailment)

            # Create pie chart with custom autopct function
            ax4.pie(sizes, explode=explode, labels=labels, colors=colors,
                    autopct=lambda pct: func(pct, sizes),  # Use the custom function
                    shadow=False, startangle=90, textprops={'fontsize': 14})

            ax4.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
            ax4.set_title('BESS Energy Distribution [kWh]')
            fig.tight_layout()

        else:
            def func(pct, allvalues):
                absolute = int(np.round(pct / 100. * np.sum(allvalues)))  # Calculate absolute value
                return f"{absolute} ({pct:.1f}%)"  # Format as "absolute (percentage)"

            sizes = [total_taken_from_pv, total_discharged_from_pv]
            labels = [ 'From PV to BESS', 'From PV to Grid']
            colors = ['steelblue', 'khaki']
            explode = (0.1, 0)  # explode the first slice (Total Curtailment)

            # Create pie chart
            ax4.pie(sizes, explode=explode, labels=labels, colors=colors,
                    autopct=lambda pct: func(pct, sizes),  # Use the custom function
                    shadow=False, startangle=90, textprops={'fontsize': 14})
            ax4.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
            ax4.set_title('BESS Energy Distribution [kWh]')
            fig.tight_layout()

        # Save the plot based on the minimize_C condition
        plt.savefig(os.path.join(self.plots_dir, "PV_View.png"))

    def POD_View(self, num_values):

        # GET VARIABLES FROM SELF
        time_steps = self.time_steps[:num_values]
        charged_energy = self.charged_energy[:num_values]
        discharged_energy = self.discharged_energy[:num_values]
        pun_values = self.PUN_timeseries[:num_values]
        soc = self.soc[:num_values]
        produced_from_pv = self.produced_from_pv[:num_values]
        taken_from_pv = self.taken_from_pv[:num_values]
        taken_from_grid = self.taken_from_grid[:num_values]
        discharged_from_pv = self.discharged_from_pv

        # EVALUATE REVENUES
        rev = - (np.array(discharged_energy) * pun_values / 1000) - (
                taken_from_grid * pun_values / 1000) + (
                      -discharged_from_pv) * pun_values / 1000
        rev_pv = -discharged_from_pv * pun_values / 1000
        rev_bess = -(np.array(discharged_energy) * pun_values / 1000) - (
                taken_from_grid * pun_values / 1000)

        rev = np.array(rev, dtype=float)
        rev_cumulative = np.cumsum(rev)

        # EVALUATE TOTAL ENERGY QUANTITIES
        total_from_pv = np.sum(taken_from_pv)
        total_from_grid = np.sum(taken_from_grid)
        total_discharged = np.sum(-discharged_energy)
        total_dicharged_pv = np.sum(-discharged_from_pv)
        curtailment = np.maximum(produced_from_pv + discharged_from_pv - taken_from_pv, 0.0)
        total_curtailment = np.sum(curtailment)

        # Creating the layout with 4 boxes using gridspec
        fig = plt.figure(figsize=(24, 12))  # Aumentato il figsize per adattarsi a 4 grafici
        gs = gridspec.GridSpec(2, 2, width_ratios=[1, 1], height_ratios=[1, 1])

        # Axis for SoC top left
        ax0 = fig.add_subplot(gs[0, 0])

        # Normalize SoC values to be in the range [0, 1] for the colormap
        norm = Normalize(vmin=min(soc*100), vmax=max(soc*100))
        cmap = matplotlib.colors.LinearSegmentedColormap.from_list("", ["lightblue", "steelblue", "darkblue"])

        # Plot SoC with gradient colored bars based on value
        for i in range(len(time_steps)):
            ax0.bar(time_steps[i], soc[i]*100, color='lightgrey')
        ax0.set_title('State of Charge (SoC)')
        ax0.set_ylabel('SoC [%]')

        # Axis for charged and discharged energy (second graph) bottom left
        ax1 = fig.add_subplot(gs[1, 0])
        ax1.fill_between(time_steps, 0, POD_power, color='lightgrey', alpha=0.3, label='POD Power')

        width = 0.4
        ax1.bar(time_steps, [1] * np.array(taken_from_grid), width=width, color='darkgreen',
                label='From Grid to BESS')
        ax1.bar(time_steps, -discharged_energy, width=width, color='darkred',
               bottom= np.array(taken_from_grid),
               label='From BESS to Grid')
        ax1.bar(time_steps, -discharged_from_pv, width= width, bottom = -discharged_energy,
                label='From PV to Grid',)

        ax1.set_ylabel('Energy [kWh]')
        ax1.set_title('POD Energy Flows')
        ax1.legend(loc='upper left')
        plt.ylim(-size * 0.6, size * 0.6)

        # Plot PUN values on the secondary axis
        ax3 = ax1.twinx()
        ax3.plot(time_steps, pun_values, color='black', label='PUN')
        ax3.set_ylabel('PUN [Euro/MWh]')
        ax3.legend(loc='upper right')
        ax1.set_xlabel('Time Window [h]')

        # Cumulative revenue axis to the right of the first graph
        ax2 = fig.add_subplot(gs[0, 1])
        ax2.plot(time_steps, rev_cumulative, color='lightgreen', label='Cumulative Revenues', alpha=1)
        ax2.set_title('Cumulative Revenues Over Time')
        ax2.fill_between(time_steps, rev_cumulative, color='green', alpha=0.3)  # Area sottesa con alpha 0.3
        ax2.set_ylabel('Cumulative Revenues [Euros]')
        ax2.legend(loc='upper left')

        # New graph for rev_pv and rev_bess (fourth graph) bottom right
        colors_bess = ['red' if total < 0 else 'limegreen' for total in rev_pv]
        ax4 = fig.add_subplot(gs[1, 1])

        def func(pct, allvalues):
            absolute = int(np.round(pct / 100. * np.sum(allvalues)))  # Calculate absolute value
            return f"{absolute} ({pct:.1f}%)"  # Format as "absolute (percentage)"

        sizes = [ total_dicharged_pv,  total_from_grid, total_discharged]
        labels = ['From PV to Grid',  'From Grid to BESS', 'From BESS to Grid']
        colors = ['khaki', 'lightblue', 'paleturquoise' ]
        explode = (0.05, 0.05,0.05)  # explode the first slice (Total Curtailment)

        # Create pie chart
        ax4.pie(sizes, explode=explode, labels=labels, colors=colors,
                autopct=lambda pct: func(pct, sizes),  # Use the custom function
                shadow=False, startangle=90, textprops={'fontsize': 14})
        ax4.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
        ax4.set_title('POD Energy Distribution [kWh]')
        fig.tight_layout()

        # Save the plot based on the minimize_C condition
        plt.savefig(os.path.join(self.plots_dir, "POD_View.png"))

    def Total_View(self, num_values):

        # GET VARIABLES FROM SELF
        time_steps = self.time_steps[:num_values]
        charged_energy = self.charged_energy[:num_values]
        discharged_energy = self.discharged_energy[:num_values]
        pun_values = self.PUN_timeseries[:num_values]
        soc = self.soc[:num_values]
        produced_from_pv = self.produced_from_pv[:num_values]
        taken_from_pv = self.taken_from_pv[:num_values]
        taken_from_grid = self.taken_from_grid[:num_values]
        discharged_from_pv = self.discharged_from_pv
        self_consumption = self.load
        from_pv_to_load = self.from_pv_to_load
        from_BESS_to_load = self.from_BESS_to_load
        shared_energy_bess = self.shared_energy_bess

        # EVALUATE REVENUES
        rev = np.array( np.abs(discharged_energy) * pun_values / 1000
               - np.abs(taken_from_grid * pun_values  / 1000)
               # + np.abs(discharged_from_pv) * pun_values / 1000
               + np.abs(shared_energy_bess) * 120 / 1000
                )

        rev = np.array(rev, dtype=float)


        # Creating the layout with 3 boxes using gridspec
        fig = plt.figure(figsize=(36, 16))  # Increased height for the new graph
        gs = gridspec.GridSpec(3, 1, height_ratios=[1, 1, 1])  # Set equal height for all rows

        # Axis for SoC top
        ax0 = fig.add_subplot(gs[0, 0])

        # Plot SoC
        norm = Normalize(vmin=min(soc * 100), vmax=max(soc * 100))
        cmap = plt.cm.get_cmap("Blues")

        for i in range(len(time_steps)):
            ax0.bar(time_steps[i], soc[i] * 100, color=cmap(norm(soc[i] * 100)))
        ax0.set_title('State of Charge (SoC)')
        ax0.set_ylabel('SoC [%]')
        plt.ylim(0, max(soc) * 100 + max(soc) * 100 * 0.08)

        # Axis for charged and discharged energy (second graph)
        ax1 = fig.add_subplot(gs[1, 0])

        total_d = []
        total_dis = np.abs(discharged_energy) + np.abs(discharged_from_pv)

        for i in range(0, len(discharged_energy), 24):
            somma = sum(total_dis[i:i + 24])
            total_d.append(somma)

        total_d = np.array(total_d)
        total_d = total_d * 30 / 1000
        total_d = np.round(total_d, 2)

        norm = (total_d - np.min(total_d)) / (np.max(total_d) - np.min(total_d))
        colors = [(1, 1 - n, 0) for n in norm]

        width = 0.4

        load = pd.to_numeric(self.load, errors='coerce')
        ax1.fill_between(time_steps, load, color='orange', alpha=0.3, label='REC Load')

        ax1.fill_between(time_steps, 0, produced_from_pv, color='lightblue', alpha=0.3, label="User's PV Production")

        ax1.bar(time_steps, -discharged_from_pv, width=width, bottom=from_pv_to_load + taken_from_pv,
                label="User's PV to Grid")

        ax1.bar(time_steps, from_pv_to_load, width=width, color="grey", label='User PV to Load')

        ax1.bar(time_steps, discharged_energy, width=width, color='darkred', bottom=np.array(taken_from_grid),
                label="User's BESS to Grid")

        ax1.bar(time_steps, taken_from_pv, width=width, color='orange', bottom=np.array(from_pv_to_load),
                label="User's PV to BESS")

        ax1.bar(time_steps, shared_energy_bess, color='cyan', width=width, bottom=from_pv_to_load+taken_from_pv+np.abs(discharged_from_pv), label='User BESS Add SE')

        ax1.bar(time_steps, [1] * np.array(taken_from_grid), width=width, color='darkgreen', label='Grid to User BESS')

        ax1.set_ylabel('Energy [kWh]')
        ax1.set_title('System Energy Flows')
        ax1.legend(loc='upper left')
        plt.ylim(min(-size * 0.6,-max(produced_from_pv)), max(size * 0.6,max(produced_from_pv)))

        # Plot PUN values on the secondary axis
        ax3 = ax1.twinx()
        ax3.plot(time_steps, pun_values, color='black', label='PUN')
        ax3.set_ylabel('PUN [Euro/MWh]')
        ax3.legend(loc='upper right')
        plt.ylim(min(pun_values) - 0.12 * min(pun_values), max(pun_values) + max(pun_values) * 0.12)
        ax1.set_xlabel('Time Window [h]')

        rev_sums = []

        for i in range(0, len(rev), 24):
            somma = sum(rev[i:i + 24])
            rev_sums.append(somma)

        rev_sums = np.array(rev_sums)
        rev_sums = rev_sums * 30
        rev_sums = np.round(rev_sums, 2)

        # New axis for revenues (third graph)
        ax2 = fig.add_subplot(gs[2, 0])

        # Create a color array based on the values in rev
        colors_rev = ['green' if r > 0 else 'red' for r in rev]

        ax2.bar(time_steps, rev, color=colors_rev, label='Revenue', width=0.4)
        ax2.set_ylabel('Revenue [Euro]')
        ax2.set_title('Revenue from Energy Transactions')
        ax2.legend(loc='upper left')
        plt.ylim(min(rev) + 0.25 * min(rev), max(rev) + 0.25 * max(rev))  # Adjust y-limits based on revenue values

        norm = (rev_sums - np.min(rev_sums)) / (np.max(rev_sums) - np.min(rev_sums))
        colors_rev = [(0.8 * (1 - n), 0.8 + 0.2 * n, 0.8 * (1 - n)) for n in norm]


        from argparser_l import weekends

        if weekends == 'True':
            ax1.set_title('System Energy Flows - Weekdays')
            ax0.set_title('State of Charge (SoC) - Weekdays')
            fig.tight_layout()
            plt.savefig(os.path.join(self.plots_dir, "Total_View.png"))
        else:
            ax1.set_title('System Energy Flows - Weekends')  # TODO: SHOULD BE WEEKENDS

        ax0.set_title('State of Charge (SoC) - Weekends')  # TODO: SHOULD BE WEEKENDS
        fig.tight_layout()
        plt.savefig(os.path.join(self.plots_dir, "Total_View.png"))

    import numpy as np
    import matplotlib.pyplot as plt
    import os

    def Dashboard(self, num_values):

        from argparser_l import size, POD_power, technology, n_cycles, soc_max, soc_min, PV_power
        import matplotlib.gridspec as gridspec

        time_steps = self.time_steps[:num_values]
        charged_energy = self.charged_energy[:num_values]
        discharged_energy = self.discharged_energy[:num_values]
        pun_values = self.PUN_timeseries[:num_values]
        soc = self.soc[:num_values]
        produced_from_pv = self.produced_from_pv[:num_values]
        taken_from_pv = self.taken_from_pv[:num_values]
        taken_from_grid = self.taken_from_grid[:num_values]
        discharged_from_pv = self.discharged_from_pv
        self_consumption = self.load
        from_pv_to_load = self.from_pv_to_load
        from_BESS_to_load = self.from_BESS_to_load
        shared_energy_bess = self.shared_energy_bess
        rec_production = self.rec_production
        rec_load = self.rec_load

        rev = np.array(np.abs(discharged_energy) * pun_values / 1000
                       - np.abs(taken_from_grid * pun_values / 1000)
                       + np.abs(shared_energy_bess) * 120 / 1000)
        rev = np.array(rev, dtype=float)

        shared_energy = np.minimum(rec_load + self.load - from_pv_to_load - from_BESS_to_load, rec_production[:, 1])
        uncovered_rec_load = (rec_load - shared_energy)

        total_rec_uncovered_load = np.sum(uncovered_rec_load) * 30
        total_shared_energy = np.sum(shared_energy) * 30
        total_shared_energy_bess = np.sum(shared_energy_bess) * 30

        uncovered_load = self.load - from_pv_to_load - from_BESS_to_load
        total_uncovered_load = np.sum(uncovered_load) * 30

        total_discharged_energy = np.abs(np.sum(discharged_energy)) * 30
        total_charged_energy = np.sum(taken_from_grid) * 30
        total_taken_from_pv = np.sum(taken_from_pv) * 30
        total_from_bess_to_load = np.sum(from_BESS_to_load) * 30

        user_shared_energy = total_shared_energy - np.sum(np.minimum(rec_load, rec_production[:, 1])) * 30

        hist_data = {
            'Disch Energy': total_discharged_energy,
            'Charg Energy': total_charged_energy,
            'Withdrawn PV': total_taken_from_pv,
            'BESS to Load': total_from_bess_to_load,
            'SE BESS': total_shared_energy_bess
        }

        # Ordino i dati per l'istogramma
        sorted_hist = dict(sorted(hist_data.items(), key=lambda item: item[1], reverse=True))
        labels_sorted = list(sorted_hist.keys())
        values_sorted = list(sorted_hist.values())

        fig = plt.figure(figsize=(14, 18))  # Adjusted height to fit 3 rows

        # New 3-row GridSpec: 3 rows, 2 columns
        # height_ratios set to roughly fit table, pies, and histogram
        gs = gridspec.GridSpec(3, 2, height_ratios=[3, 5, 7], hspace=0.3, wspace=0.3)

        # Table on top spanning both columns (row 0, both columns)
        ax_table = fig.add_subplot(gs[0, :])
        ax_table.axis('off')

        table_data = [
            ['User Battery Size (kWh)', f'{size:.1f}'],
            ['User POD (kW)', f'{POD_power:.1f}'],
            ['BESS Technology', technology],
            ['DoD', f'{soc_min * 100}% - {soc_max * 100}%'],
            ['User PV Size (kW)', f'{PV_power:.1f}']
        ]

        table = ax_table.table(cellText=table_data,
                               colLabels=['Parameter', 'Value'],
                               cellLoc='center',
                               colLoc='center',
                               loc='center')

        table.auto_set_font_size(False)
        table.set_fontsize(14)
        table.scale(1, 2)

        for key, cell in table.get_celld().items():
            row, col = key
            if row == 0:
                cell.get_text().set_fontweight('bold')

        adjusted_uncovered_load = total_uncovered_load - np.sum(user_shared_energy)
        if adjusted_uncovered_load < 0:
            adjusted_uncovered_load = 0  # avoid negative values for pie chart slice

        # Pie charts side-by-side in row 1
        ax_pie1 = fig.add_subplot(gs[1, 0])
        colors1 = ['gainsboro', 'yellowgreen', 'cyan']
        sizes1 = [total_rec_uncovered_load, total_shared_energy, total_shared_energy_bess]
        labels1 = ['REC Uncovered Load', 'Shared Energy', 'Additional SE BESS']
        wedges1, texts1, autotexts1 = ax_pie1.pie(sizes1, labels=labels1, autopct='%1.1f%%', startangle=140,
                                                      colors=colors1,
                                                      textprops={'fontsize': 15},
                                                      wedgeprops={'alpha': 0.85},
                                                  labeldistance=1.15, pctdistance=0.85)
        ax_pie1.set_title('REC Dashboard', fontsize=20, weight='bold')
        for autotext in autotexts1:
            autotext.set_color('black')

        # Add numerical values below each label in pie1
        for text, value in zip(autotexts1, sizes1):
            x, y = text.get_position()
            ax_pie1.text(x, y - 0.1, f'{value:,.0f} kWh', ha='center', va='top', fontsize=10, color='black')

        ax_pie2 = fig.add_subplot(gs[1, 1])
        colors2 = ['steelblue', 'orange', 'yellowgreen', 'gainsboro', 'cyan']
        sizes2 = [np.sum(from_pv_to_load) * 30, np.sum(from_BESS_to_load) * 30, np.sum(user_shared_energy),
                  adjusted_uncovered_load, total_shared_energy_bess]
        labels2 = ['PV to Load (SC)', 'BESS to Load (SC)', 'User SE', 'Uncovered Load', 'BESS SE']
        wedges2, texts2, autotexts2 = ax_pie2.pie(sizes2, labels=labels2, autopct='%1.1f%%', startangle=140,
                                                      colors=colors2,
                                                      textprops={'fontsize': 15},
                                                      wedgeprops={'alpha': 0.85},
                                                  labeldistance=1.15, pctdistance=0.85)
        ax_pie2.set_title('User Dashboard', fontsize=20, weight='bold')
        for autotext in autotexts2:
            autotext.set_color('black')

        # Add numerical values below each label in pie2
        for text, value in zip(autotexts2, sizes2):
            x, y = text.get_position()
            ax_pie2.text(x, y - 0.1, f'{value:,.0f} kWh', ha='center', va='top', fontsize=10, color='black')

        # Histogram spanning both columns in row 2
        # Add 'Produced from PV' and 'User Shared Energy' to histogram data
        hist_data.update({
            'User PV': np.abs(np.sum(produced_from_pv)) * 30,
            'User SE': np.abs(np.sum(user_shared_energy) * 30)
        })

        sorted_hist = dict(sorted(hist_data.items(), key=lambda item: item[1], reverse=True))
        labels_sorted = list(sorted_hist.keys())
        values_sorted = list(sorted_hist.values())

        ax_hist = fig.add_subplot(gs[2, :])
        bars = ax_hist.bar(labels_sorted, values_sorted,
                           color=[{
                                      'Disch Energy': 'darkorchid',
                                      'Charg Energy': 'darkgreen',
                                      'Withdrawn PV': 'indigo',
                                      'BESS to Load': 'orange',
                                      'SE BESS': 'cyan',
                                      'User PV': 'mediumseagreen',
                                      'User SE': 'yellowgreen'
                                  }[label] for label in labels_sorted])
        max_height = max(values_sorted)
        ax_hist.set_ylim(0, max_height * 1.25)

        for bar in bars:
            height = bar.get_height()
            ax_hist.annotate(f'{height:,.0f} kWh',
                             xy=(bar.get_x() + bar.get_width() / 2, height),
                             xytext=(0, 6),
                             textcoords='offset points',
                             ha='center', va='bottom',
                             fontsize=13, color='black')

        ax_hist.set_ylabel('Energy [kWh]', fontsize=13)
        ax_hist.set_title('Energy Flows', fontsize=16, pad=15, weight="bold")
        ax_hist.tick_params(axis='x', labelsize=12)
        ax_hist.tick_params(axis='y', labelsize=12)
        ax_hist.grid(False)
        ax_hist.axhline(0, color='black', linewidth=1)
        ax_hist.legend(bars, labels_sorted, loc='upper right', fontsize=15, frameon=True)

        ax_hist.margins(x=0.15)

        plt.tight_layout()

        plt.savefig(os.path.join(self.plots_dir, "Dashboard.png"), dpi=150, bbox_inches='tight')
        plt.close(fig)


    def USER_View(self, num_values):

        # GET VARIABLES FROM SELF
        time_steps = self.time_steps[:num_values]
        charged_energy = self.charged_energy[:num_values]
        discharged_energy = self.discharged_energy[:num_values]
        pun_values = self.PUN_timeseries[:num_values]
        soc = self.soc[:num_values]
        produced_from_pv = self.produced_from_pv[:num_values]
        taken_from_pv = self.taken_from_pv[:num_values]
        taken_from_grid = self.taken_from_grid[:num_values]
        discharged_from_pv = self.discharged_from_pv
        self_consumption = self.load
        from_pv_to_load = self.from_pv_to_load
        from_BESS_to_load = self.from_BESS_to_load
        shared_energy_bess = self.shared_energy_bess

        # SET MONTH NAMES
        month_names = ["January", "February", "March", "April", "May", "June",
            "July", "August", "September", "October", "November", "December"]

        # Creating the layout with 3 boxes using gridspec
        fig = plt.figure(figsize=(28, 10))  # Increased height for the new graph
        gs = gridspec.GridSpec(2, 1, height_ratios=[1, 1])  # Set equal height for all rows

        # Axis for SoC top
        ax0 = fig.add_subplot(gs[0, 0])

        # Plot SoC
        norm = Normalize(vmin=min(soc * 100), vmax=max(soc * 100))
        cmap = plt.cm.get_cmap("Blues")
        for i in range(12):
            start = i * 24 - 1
            end = (i + 1) * 24 - 1
            plt.axvline(x=i * 24 - 1, ls='--', color="black")
            ax0.text(11.5 + 23.92 * i, max(soc) * 100 + max(soc) * 100 * 0.04, f'{month_names[i]}', color="black",
                     fontsize=15, horizontalalignment='center')

        for i in range(len(time_steps)):
            ax0.bar(time_steps[i], soc[i] * 100, color=cmap(norm(soc[i] * 100)))
        ax0.set_title('State of Charge (SoC)')
        ax0.set_ylabel('SoC [%]')
        plt.ylim(0, max(soc) * 100 + max(soc) * 100 * 0.08)

        # Axis for charged and discharged energy (second graph)
        ax1 = fig.add_subplot(gs[1, 0])

        total_d = []
        total_dis = np.abs(discharged_energy) + np.abs(discharged_from_pv)

        for i in range(0, len(discharged_energy), 24):
            somma = sum(total_dis[i:i + 24])
            total_d.append(somma)

        total_d = np.array(total_d)
        total_d = total_d * 30 / 1000
        total_d = np.round(total_d, 2)

        norm = (total_d - np.min(total_d)) / (np.max(total_d) - np.min(total_d))
        colors = [(1, 1 - n, 0) for n in norm]

        width = 0.4

        load = pd.to_numeric(self.load, errors='coerce')
        ax1.fill_between(time_steps, load, color='orange', alpha=0.3, label='USER Load')

        ax1.fill_between(time_steps, 0, produced_from_pv, color='lightblue', alpha=0.3, label="PV Production")

        ax1.bar(time_steps, -discharged_from_pv, width=width, bottom=from_pv_to_load + taken_from_pv,
                label="PV to Grid")

        ax1.bar(time_steps, from_pv_to_load, width=width, color="grey", label='User PV to Load')

        ax1.bar(time_steps, from_BESS_to_load, width=width, color='darkorange', bottom=np.array(from_pv_to_load),
                label="BESS to Load")

        ax1.bar(time_steps, discharged_energy, width=width, color='darkred', bottom=np.array(taken_from_grid),
                label="BESS to Grid")

        ax1.bar(time_steps, taken_from_pv, width=width, color='orange', bottom=np.array(from_pv_to_load + from_BESS_to_load),
                label="PV to BESS")

        ax1.bar(time_steps, shared_energy_bess, color='cyan', width=width, bottom=from_pv_to_load+taken_from_pv+np.abs(discharged_from_pv), label="Shared Energy")

        ax1.bar(time_steps, [1] * np.array(taken_from_grid), width=width, color='darkgreen', label='Grid to BESS')

        ax1.set_ylabel('Energy [kWh]')
        ax1.set_title('System Energy Flows')
        ax1.legend(loc='upper left')
        plt.ylim(min(-size * 0.6,-max(produced_from_pv)), max(size * 0.6,max(produced_from_pv)))

        # Plot PUN values on the secondary axis
        ax3 = ax1.twinx()
        ax3.plot(time_steps, pun_values, color='black', label='PUN', alpha=0.5)
        ax3.set_ylabel('PUN [Euro/MWh]')
        ax3.legend(loc='upper right')
        plt.ylim(min(pun_values) - 0.12 * min(pun_values), max(pun_values) + max(pun_values) * 0.12)
        ax1.set_xlabel('Time Window [h]')

        from argparser_l import weekends

        if weekends == 'True':
            ax1.set_title('System Energy Flows - Weekdays')
            ax0.set_title('State of Charge (SoC) - Weekdays')
            fig.tight_layout()
            plt.savefig(os.path.join(self.plots_dir, "Total_View.png"))
        else:
            ax1.set_title('System Energy Flows')

        ax0.set_title('State of Charge (SoC)')
        fig.tight_layout()
        plt.savefig(os.path.join(self.plots_dir, "User_View.png"))

    def REC_View(self, num_values):

        # GET VARIABLES FROM SELF
        time_steps = self.time_steps[:num_values]
        charged_energy = self.charged_energy[:num_values]
        discharged_energy = self.discharged_energy[:num_values]
        pun_values = self.PUN_timeseries[:num_values]
        soc = self.soc[:num_values]
        produced_from_pv = self.produced_from_pv[:num_values]
        taken_from_pv = self.taken_from_pv[:num_values]
        taken_from_grid = self.taken_from_grid[:num_values]
        discharged_from_pv = self.discharged_from_pv
        self_consumption = self.load
        from_pv_to_load = self.from_pv_to_load
        from_BESS_to_load = self.from_BESS_to_load
        shared_energy_bess = self.shared_energy_bess
        rec_production = self.rec_production
        rec_load = self.rec_load

        # SET MONTH NAMES
        month_names = ["January", "February", "March", "April", "May", "June",
            "July", "August", "September", "October", "November", "December"]

        # Creating the layout with 3 boxes using gridspec
        fig = plt.figure(figsize=(28, 10))  # Increased height for the new graph
        gs = gridspec.GridSpec(2, 1, height_ratios=[1, 1])  # Set equal height for all rows

        # Axis for SoC top
        ax0 = fig.add_subplot(gs[0, 0])

        # Plot SoC
        norm = Normalize(vmin=min(soc * 100), vmax=max(soc * 100))
        cmap = plt.cm.get_cmap("Blues")
        for i in range(12):
            start = i * 24 - 1
            end = (i + 1) * 24 - 1
            plt.axvline(x=i * 24 - 1, ls='--', color="black")
            ax0.text(11.5 + 23.92 * i, max(soc) * 100 + max(soc) * 100 * 0.04, f'{month_names[i]}', color="black",
                     fontsize=15, horizontalalignment='center')

        for i in range(len(time_steps)):
            ax0.bar(time_steps[i], soc[i] * 100, color=cmap(norm(soc[i] * 100)))
        ax0.set_title('State of Charge (SoC)')
        ax0.set_ylabel('SoC [%]')
        plt.ylim(0, max(soc) * 100 + max(soc) * 100 * 0.08)

        # Axis for charged and discharged energy (second graph)
        ax1 = fig.add_subplot(gs[1, 0])

        width = 0.4

        ax1.fill_between(time_steps, np.array(rec_load), color='lightgreen', alpha=0.3, label='REC Load')

        ax1.fill_between(time_steps,  np.array(rec_production[:,1]), color='lightblue', alpha=0.3, label="REC Production")

        shared_energy = np.minimum(np.array(rec_load), np.array(rec_production[:,1]))

        #ax1.fill_between(time_steps, shared_energy_bess + shared_energy, color='indigo', alpha=0.3, label="Additional SE")

        ax1.fill_between(time_steps, shared_energy, color='plum', alpha=0.3, label="Shared Energy")

        ax1.set_ylabel('Energy [kWh]')
        ax1.set_title('System Energy Flows')
        ax1.legend(loc='upper left')
        plt.ylim(0, max(rec_production[:,1]))

        from argparser_l import weekends

        if weekends == 'True':
            ax1.set_title('System Energy Flows - Weekdays')
            ax0.set_title('State of Charge (SoC) - Weekdays')
            fig.tight_layout()
            plt.savefig(os.path.join(self.plots_dir, "Total_View.png"))
        else:
            ax1.set_title('System Energy Flows')

        ax0.set_title('State of Charge (SoC)')
        fig.tight_layout()
        plt.savefig(os.path.join(self.plots_dir, "REC_View.png"))


    def Total_View_cycles(self, num_values, n_cycles):

        # Create a cycles vector that matches the length of the time window
        cycles = np.linspace(n_cycles[0], n_cycles[-1], num_values)
        charged_energy = self.charged_energy[:num_values]
        discharged_energy = self.discharged_energy[:num_values]
        pun_values = self.PUN_timeseries[:num_values]
        soc = self.soc[:num_values]
        produced_from_pv = self.produced_from_pv[:num_values]
        taken_from_pv = self.taken_from_pv[:num_values]  # Energy from PV
        taken_from_grid = self.taken_from_grid[:num_values]  # Energy from the grid
        discharged_from_pv = self.discharged_from_pv

        # EVALUATE REVENUES
        rev = - (np.array(discharged_energy) * pun_values / 1000) - (
                taken_from_grid * pun_values / 1000) + (
                  -discharged_from_pv) * pun_values / 1000
        rev_pv = -discharged_from_pv * pun_values / 1000
        rev_bess = -(np.array(discharged_energy) * pun_values / 1000) - (
                taken_from_grid * pun_values / 1000)
        rev = np.array(rev, dtype=float)
        rev_cumulative = np.cumsum(rev)

        # EVALUATE TOTAL ENTITIES
        total_from_pv = np.sum(taken_from_pv)
        total_from_grid = np.sum(taken_from_grid)
        total_discharged = np.sum(-discharged_energy)
        total_discharged_pv = np.sum(-discharged_from_pv)
        curtailment = np.maximum(produced_from_pv + discharged_from_pv - taken_from_pv, 0.0)
        total_curtailment = np.sum(curtailment)

        # Create layout with 4 boxes using gridspec
        fig = plt.figure(figsize=(24, 12))
        gs = gridspec.GridSpec(2, 2, width_ratios=[1, 1], height_ratios=[1, 1])

        # Axes for SoC in top left
        ax0 = fig.add_subplot(gs[0, 0])

        # Normalize SoC values to be in the range [0, 1] for the colormap
        norm = Normalize(vmin=min(soc * 100), vmax=max(soc * 100))
        cmap = matplotlib.colors.LinearSegmentedColormap.from_list("", ["lightblue", "steelblue", "darkblue"])
        width = 0.1

        # Plot SoC with gradient colored bars based on value
        for i in range(len(cycles)):
            ax0.bar(cycles[i], soc[i] * 100, color=cmap(norm(soc[i])), width=width)
        ax0.set_title('State of Charge (SoC)')
        ax0.set_ylabel('SoC [%]')
        width = 0.05

        # Axes for energy charged and discharged (second graph) in bottom left
        ax1 = fig.add_subplot(gs[1, 0])

        # Add area under 'produced_from_pv' in a lighter blue behind the bars
        ax1.fill_between(cycles, 0, produced_from_pv, color='lightblue', alpha=0.3, label='Produced from PV')
        ax1.bar(cycles, [1] * np.array(taken_from_grid), width=width, color='darkgreen',
                label='From Grid to BESS')
        ax1.bar(cycles, taken_from_pv, color='darkblue', bottom=-discharged_from_pv, width=width,
                label='From PV to BESS')

        ax1.bar(cycles, discharged_energy, width=width, color='darkred',
                bottom=np.array(taken_from_grid),
                label='From BESS to Grid')
        ax1.bar(cycles, -discharged_from_pv, width=width,
                label='From PV to Grid')

        ax1.set_ylabel('Energy [kWh]')
        ax1.set_title('System Energy Flows')
        ax1.legend(loc='upper left')
        plt.ylim(-size * 0.6, size * 0.6)

        # Plot PUN values on the secondary axis
        ax3 = ax1.twinx()
        ax3.plot(cycles, pun_values, color='black', label='PUN')
        ax3.set_ylabel('PUN [Euro/MWh]')
        ax3.legend(loc='upper right')
        ax1.set_xlabel('Cycles')

        # Axes for cumulative revenues to the right of the first graph
        ax2 = fig.add_subplot(gs[0, 1])  # Use only the first row in the right column
        ax2.plot(cycles, rev_cumulative, color='lightgreen', label='Cumulative Revenues', alpha=1)
        ax2.set_title('Cumulative Revenues Over Time')
        ax2.fill_between(cycles, rev_cumulative, color='green', alpha=0.3)  # Area under with alpha 0.3
        ax2.set_ylabel('Cumulative Revenues [Euros]')
        ax2.legend(loc='upper left')

        # New graph for rev_pv and rev_bess (fourth graph) in bottom right
        colors_bess = ['red' if total < 0 else 'limegreen' for total in rev_pv]
        ax4 = fig.add_subplot(gs[1, 1])

        if total_curtailment != 0.0:
            def func(pct, allvalues):
                absolute = int(np.round(pct / 100. * np.sum(allvalues)))  # Calculate absolute value
                return f"{absolute} ({pct:.1f}%)"  # Format as "absolute (percentage)"

            sizes = [total_curtailment, total_discharged_pv, total_from_pv, total_from_grid, total_discharged]
            labels = ['Curtailment', 'From PV to Grid', 'From PV to BESS', 'From Grid to BESS', 'From BESS to Grid']
            colors = ['Orange', 'khaki', 'steelblue', 'lightblue', 'paleturquoise']
            explode = (0.05, 0.05, 0.05, 0.05, 0.05)  # explode the first slice (Total Curtailment)

            # Create pie chart
            ax4.pie(sizes, explode=explode, labels=labels, colors=colors,
                    autopct=lambda pct: func(pct, sizes),  # Use the custom function
                    shadow=False, startangle=90, textprops={'fontsize': 14})
            ax4.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
            ax4.set_title('System Total Energy Distribution [kWh]')

        else:
            def func(pct, allvalues):
                absolute = int(np.round(pct / 100. * np.sum(allvalues)))  # Calculate absolute value
                return f"{absolute} ({pct:.1f}%)"  # Format as "absolute (percentage)"

            sizes = [total_discharged_pv, total_from_pv, total_from_grid, total_discharged]
            labels = ['From PV to Grid', 'From PV to BESS', 'From Grid to BESS', 'From BESS to Grid']
            colors = ['goldenrod', 'steelblue', 'lightblue', 'paleturquoise']
            explode = (0.05, 0.05, 0.05, 0.05)  # explode the first slice (Total Curtailment)

            # Create pie chart
            ax4.pie(sizes, explode=explode, labels=labels, colors=colors,
                    autopct=lambda pct: func(pct, sizes),  # Use the custom function
                    shadow=False, startangle=90, textprops={'fontsize': 14})
            ax4.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
            ax4.set_title('System Total Energy Distribution [kWh]')
        fig.tight_layout()

        # Save the plot based on the minimize_C condition
        plt.savefig(os.path.join(self.plots_dir, "Total_View_cycles.png"))

    def Total_View_Monthly(self, num_values, rows_per_day=24):
        # Extract relevant data
        time_steps = self.time_steps[:num_values]
        charged_energy = self.charged_energy[:num_values]
        discharged_energy = self.discharged_energy[:num_values]
        pun_values = self.PUN_timeseries[:num_values]
        soc = self.soc[:num_values]
        produced_from_pv = self.produced_from_pv[:num_values]
        taken_from_pv = self.taken_from_pv[:num_values]  # Energy from PV
        taken_from_grid = self.taken_from_grid[:num_values]  # Energy from the grid
        discharged_from_pv = self.discharged_from_pv

        # Create a figure with 12 subplots (one for each month)
        fig = plt.figure(figsize=(24, 18))
        gs = gridspec.GridSpec(3, 4)  # 3 rows and 4 columns for 12 months

        # Calculate the row index for the 15th day of each month
        rows_per_month = 30 * rows_per_day  # Assuming 30 days per month
        day_15_indices = [(14 * rows_per_day) + (month - 1) * rows_per_month for month in range(1, 13)]
        month_names = [
            "January", "February", "March", "April",
            "May", "June", "July", "August",
            "September", "October", "November", "December"
        ]

        # Loop through each month and plot
        for month, index in enumerate(day_15_indices):
            if index < num_values:  # Ensure index is within bounds
                # Filter data for the 15th day of the current month
                filtered_time_steps = time_steps[index:index + rows_per_day]
                filtered_soc = soc[index:index + rows_per_day]
                filtered_produced_from_pv = produced_from_pv[index:index + rows_per_day]
                filtered_taken_from_pv = taken_from_pv[index:index + rows_per_day]
                filtered_taken_from_grid = taken_from_grid[index:index + rows_per_day]
                filtered_discharged_energy = discharged_energy[index:index + rows_per_day]
                filtered_pun_values = pun_values[index:index + rows_per_day]
                filtered_discharged_from_pv = discharged_from_pv[index:index + rows_per_day]

                # Determine the subplot position
                ax = fig.add_subplot(gs[month // 4, month % 4])

                # Normalize SoC values to be in the range [0, 1] for the colormap
                norm = Normalize(vmin=min(filtered_soc * 100), vmax=max(filtered_soc * 100))
                cmap = plt.cm.viridis  # Choose any colormap

                # Plot SoC with gradient colored bars based on value
                ax.bar(filtered_time_steps, -filtered_discharged_from_pv, width=0.4,
                        label='From PV to Grid', bottom= filtered_taken_from_pv)
                ax.set_title(month_names[month])
                ax.set_ylabel('SoC [%]')
                ax.set_xlabel('Time [h]')

                # Plot energy flows in the same subplot
                ax.fill_between(filtered_time_steps, 0, filtered_produced_from_pv, color='lightblue', alpha=0.3,
                                label='Produced from PV')
                ax.bar(filtered_time_steps, filtered_taken_from_grid, width=0.4, color='darkgreen',
                       label='From Grid to BESS')
                ax.bar(filtered_time_steps, filtered_taken_from_pv, width=0.4, color='darkblue',
                       bottom=-filtered_discharged_energy, label='From PV to BESS')
                ax.bar(filtered_time_steps, filtered_discharged_energy, width=0.4, color='darkred',
                       label='From BESS to Grid')

                # Plot PUN values on a secondary axis
                ax2 = ax.twinx()
                ax2.plot(filtered_time_steps, filtered_pun_values, color='black', label='PUN', linestyle='--')
                ax2.set_ylabel('PUN [Euro/MWh]')

                # Add legends
                ax.legend(loc='upper left')
                ax2.legend(loc='upper right')

        # Save the figure to the Plots/Results directory
        plot_dir = '../../Plots/Results'
        if not os.path.exists(plot_dir):
            os.makedirs(plot_dir)
        plt.savefig(os.path.join(plot_dir, 'Monthly_Results.png'), dpi=300, bbox_inches='tight')

    def Total_View_Monthly_SoC(self, num_values, rows_per_day=24):
        # Extract relevant data
        time_steps = self.time_steps[:num_values]
        soc = self.soc[:num_values]

        # Create a figure with 12 subplots (one for each month)
        fig = plt.figure(figsize=(24, 18))
        gs = gridspec.GridSpec(3, 4)  # 3 rows and 4 columns for 12 months

        # List of month names
        month_names = [
            "January", "February", "March", "April",
            "May", "June", "July", "August",
            "September", "October", "November", "December"
        ]

        # Calculate the row index for the 15th day of each month
        rows_per_month = 30 * rows_per_day  # Assuming 30 days per month
        day_15_indices = [(14 * rows_per_day) + (month - 1) * rows_per_month for month in range(1, 13)]

        # Normalize SoC values to be in the range [0, 1] for the colormap
        norm = Normalize(vmin=min(soc * 100), vmax=max(soc * 100))
        cmap = matplotlib.colors.LinearSegmentedColormap.from_list("", ["lightblue", "steelblue", "darkblue"])

        # Loop through each month and plot
        for month, index in enumerate(day_15_indices):
            if index < num_values:  # Ensure index is within bounds
                # Filter data for the 15th day of the current month
                filtered_time_steps = time_steps[index:index + rows_per_day]
                filtered_soc = soc[index:index + rows_per_day]

                # Determine the subplot position
                ax = fig.add_subplot(gs[month // 4, month % 4])

                # Plot SoC with gradient colored bars based on value
                for i in range(len(filtered_time_steps)):
                    ax.bar(filtered_time_steps[i], filtered_soc[i] * 100, color=cmap(norm(filtered_soc[i])))

                ax.set_title(month_names[month])  # Set title to month name
                ax.set_ylabel('SoC [%]')
                ax.set_xlabel('Time [h]')

        # Save the figure to the Plots/Results directory
        plot_dir = '../../Plots/Results'
        if not os.path.exists(plot_dir):
            os.makedirs(plot_dir)
        plt.savefig(os.path.join(plot_dir, 'Monthly_SoC.png'), dpi=300, bbox_inches='tight')

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
        plt.savefig("Plots/General/charge_rate.png")
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
        if not os.path.exists("../../Plots"):
            os.makedirs("Plots")
        plt.savefig("Plots/General/interpolated_functions.png")

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
        plt.savefig("Plots/General/charge_rate.png")
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
        plt.savefig("Plots/General/disc_rate.png")


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
        plt.savefig('Plots/Results/Short Simulation/total_convergence.png')

    @staticmethod
    def PUN_plot(PUN_timeseries):

        # Line plot dei valori PUN (terza colonna del DataFrame)
        pun_values = PUN_timeseries  # Estrazione della terza colonna (indice 2)
        plt.figure(figsize=(12, 8))
        plt.plot(pun_values, marker='o', color='b')
        plt.title('PUN Values')
        plt.xlabel('Time step')
        plt.ylabel('PUN Value')
        plt.savefig(os.path.join("Plots/General", "PUN.png"))
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
            output_dir = 'Plots/Results/Short Simulation'

            if not os.path.exists(output_dir):
                os.makedirs(output_dir)

            # Save the figure in the "Plots" folder
            output_path = os.path.join(output_dir, f'convergence_{fig_num + 1}.png')
            fig.savefig(output_path)


    @staticmethod
    def plot_alpha_vs_timewindow(time_window, alpha_values, PUN_timeseries, new_alpha_values):

        """
        Creates a bar plot of the values of alpha and new_alpha_values, and a line plot of PUN_timeseries from 1 to time_window
        with different y-axes scales.

        Parameters:
        - time_window (int): The length of the time window.
        - alpha_values (array-like): The alpha values to plot. The length of alpha should be equal to time_window.
        - PUN_timeseries (array-like): The PUN_timeseries values to plot. Should have the same length as alpha.
        - new_alpha_values (array-like): The new alpha values to plot alongside the existing ones.

        Returns:
        - None: Displays the plot.
        """

        if len(alpha_values) != time_window or len(PUN_timeseries) != time_window or len(
                new_alpha_values) != time_window:
            raise ValueError("The lengths of alpha, new_alpha_values, and PUN_timeseries must be equal to time_window")

        # Creating the time vector from 1 to time_window
        time_vector = range(1, time_window + 1)

        # Creating the figure and the first subplot for alpha (bar plot)
        fig, ax1 = plt.subplots(figsize=(10, 6))
        bar_width = 0.4  # Width of the bars

        # Positions for the bars
        bar_positions1 = np.arange(time_window - 1)
        bar_positions2 = bar_positions1 + bar_width
        ax1.bar(bar_positions1, np.abs(alpha_values[:-1]), width=bar_width, color='orange', alpha=0.5,
                label='C-rate')
        ax1.bar(bar_positions2, np.abs(new_alpha_values[:-1]), width=bar_width, color='green', alpha=0.5,
                label='Nominal C-rate')
        ax1.set_xlabel('Time')
        ax1.set_ylabel('C-rate', color='black')
        ax1.tick_params(axis='y', labelcolor='black')
        ax1.grid(False)

        # Creating the second subplot for PUN_timeseries (line plot)
        ax2 = ax1.twinx()
        ax2.plot(time_vector, PUN_timeseries, linestyle='-', color='black', label='PUN_timeseries', alpha=0.8)
        ax2.set_ylabel('PUN_timeseries Values', color='black')
        ax2.tick_params(axis='y', labelcolor='black')

        # Adding title and legend
        plt.title('C-rate and PUN_timeseries Values over Time Window')
        ax1.legend(loc='upper left')
        ax2.legend(loc='upper right')

        # Saving the plot to a file
        output_dir = 'Plots/Results/Short Simulation'
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, 'C_rate.png')
        plt.savefig(output_path)

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










