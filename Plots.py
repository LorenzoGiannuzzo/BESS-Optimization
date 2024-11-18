"""

BESS Optimization using NSGA-III Algorithm

    __author__ = "Lorenzo Giannuzzo"
    __maintainer__ = "Lorenzo Giannuzzo"
    __email__ = "lorenzo.giannuzzo@polito.it"
    __status__ = "in progress"
    __version__ = "v0.2.1"
    __license__ = "MIT"

Last Update of current code: 06/11/2024 - 17:49

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
from argparser import minimize_C, size, POD_power

matplotlib.use('Agg')


# DEFINE ENERGY PLOT CLASS

class EnergyPlots:

    def __init__(self, time_window, soc, charged_energy, discharged_energy, PUN_timeseries, taken_from_grid, taken_from_pv, produced_from_pv,discharged_from_pv):

        self.time_window = time_window
        self.soc = soc
        self.charged_energy = charged_energy
        self.discharged_energy = discharged_energy
        self.PUN_timeseries = PUN_timeseries
        self.time_steps = np.arange(time_window)
        self.taken_from_grid = taken_from_grid
        self.taken_from_pv = taken_from_pv
        self.produced_from_pv = produced_from_pv
        self.discharged_from_pv = discharged_from_pv

        #TODO This shouldn't be there because of minimize_C delete

        if minimize_C:

            self.plots_dir = "Plots/minimize_C_rate"

            if not os.path.exists(self.plots_dir):

                os.makedirs(self.plots_dir)
        else:

            self.plots_dir = "Plots/optimal_C_rate"

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

        # TODO This shouldn't be there because of minimize_C delete
        if minimize_C:

            plt.savefig(os.path.join(self.plots_dir, "SoC_minC.png"))

        else:

            plt.savefig(os.path.join(self.plots_dir, "SoC.png"))

    # DEFINE CHARGED ENERGY PLOT FUNCTION

    def plot_charged_energy(self):

        # SET FIG SIZE

        plt.figure(figsize=(12, 8))

        plt.bar(self.time_steps, self.charged_energy, color='limegreen')
        plt.title('Charged Energy')
        plt.xlabel('Time Window [h]')
        plt.ylabel('Charged Energy [kWh]')

        # TODO This shouldn't be there because of minimize_C delete
        if minimize_C:

            plt.savefig(os.path.join(self.plots_dir, "Charged_Energy_minC.png"))

        else:

            plt.savefig(os.path.join(self.plots_dir, "Charged_Energy.png"))


    # DEFINE DISCHARGED ENERGY FUNCTION

    def plot_discharged_energy(self):

        plt.figure(figsize=(12, 8))

        plt.bar(self.time_steps, self.discharged_energy, color='darkred')
        plt.title('Discharged Energy')
        plt.xlabel('Time Window [h]')
        plt.ylabel('Discharged Energy [kWh]')

        if minimize_C:

            plt.savefig(os.path.join(self.plots_dir, "Disc_Energy_minC.png"))

        else:

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

        # TODO Rembemer to fix this

        if minimize_C:

            plt.savefig(os.path.join(self.plots_dir, "BESS_View_minC.png"))

        else:

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

        # Creazione del layout con 4 box usando gridspec

        fig = plt.figure(figsize=(24, 12))
        gs = gridspec.GridSpec(2, 2, width_ratios=[1, 1], height_ratios=[1, 1])

        # Asse per SoC in alto a sinistra

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

        # Asse per l'energia caricata e scaricata (secondo grafico) in basso a sinistra

        ax1 = fig.add_subplot(gs[1, 0])

        # Aggiungi l'area sottesa per 'produced_from_pv' in un giallo più acceso e dietro le barre

        ax1.fill_between(time_steps, 0, produced_from_pv, color='lightblue', alpha=0.3, label='Produced from PV')

        width = 0.4

        ax1.bar(time_steps, taken_from_pv, color='darkblue', bottom=-discharged_from_pv,
                label='From PV to BESS', width=width)

        ax1.bar(time_steps, -discharged_from_pv,
                label='From PV to Grid', width=width)

        ax1.set_ylabel('Energy [kWh]')
        ax1.set_title('PV Energy Flows')
        ax1.legend(loc='upper left')
        plt.ylim(-size * 0.6, size * 0.6)

        # Plot PUN values on the secondary axis

        ax3 = ax1.twinx()
        ax3.plot(time_steps, pun_values, color='black', label='PUN')
        ax3.set_ylabel('PUN [Euro/MWh]')
        ax3.legend(loc='upper right')

        ax1.set_xlabel('Time Window [h]')

        # Asse per la cumulata dei ricavi a destra del primo grafico

        ax2 = fig.add_subplot(gs[0, 1])  # Usa solo la prima riga nella colonna di destra
        ax2.plot(time_steps, -rev_cumulative, color='lightgreen', label='Cumulative Revenues', alpha=1)
        ax2.set_title('Cumulative Revenues Over Time')
        ax2.fill_between(time_steps, -rev_cumulative, color='green', alpha=0.3)  # Area sottesa con alpha 0.3
        ax2.set_ylabel('Cumulative PV Revenues [Euros]')
        ax2.legend(loc='upper left')

        # Nuovo grafico per il pie chart in basso a destra
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
        if minimize_C:
            plt.savefig(os.path.join(self.plots_dir, "PV_View_minC.png"))
        else:
            plt.savefig(os.path.join(self.plots_dir, "PV_View.png"))


    def POD_View(self, num_values):
        time_steps = self.time_steps[:num_values]
        charged_energy = self.charged_energy[:num_values]
        discharged_energy = self.discharged_energy[:num_values]
        pun_values = self.PUN_timeseries[:num_values]
        soc = self.soc[:num_values]
        produced_from_pv = self.produced_from_pv[:num_values]
        taken_from_pv = self.taken_from_pv[:num_values]  # Energia da PV
        taken_from_grid = self.taken_from_grid[:num_values]  # Energia dalla rete
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
        total_dicharged_pv = np.sum(-discharged_from_pv)

        curtailment = np.maximum(produced_from_pv + discharged_from_pv - taken_from_pv, 0.0)
        total_curtailment = np.sum(curtailment)

        # Creazione del layout con 4 box usando gridspec
        fig = plt.figure(figsize=(24, 12))  # Aumentato il figsize per adattarsi a 4 grafici
        gs = gridspec.GridSpec(2, 2, width_ratios=[1, 1], height_ratios=[1, 1])

        # Asse per SoC in alto a sinistra
        ax0 = fig.add_subplot(gs[0, 0])

        # Normalize SoC values to be in the range [0, 1] for the colormap
        norm = Normalize(vmin=min(soc*100), vmax=max(soc*100))
        cmap = matplotlib.colors.LinearSegmentedColormap.from_list("", ["lightblue", "steelblue", "darkblue"])

        # Plot SoC with gradient colored bars based on value
        for i in range(len(time_steps)):
            ax0.bar(time_steps[i], soc[i]*100, color='lightgrey')
        ax0.set_title('State of Charge (SoC)')
        ax0.set_ylabel('SoC [%]')

        # Asse per l'energia caricata e scaricata (secondo grafico) in basso a sinistra
        ax1 = fig.add_subplot(gs[1, 0])

        # Aggiungi l'area sottesa per 'produced_from_pv' in un giallo più acceso e dietro le barre
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

        # Asse per la cumulata dei ricavi a destra del primo grafico
        ax2 = fig.add_subplot(gs[0, 1])  # Usa solo la prima riga nella colonna di destra
        ax2.plot(time_steps, rev_cumulative, color='lightgreen', label='Cumulative Revenues', alpha=1)
        ax2.set_title('Cumulative Revenues Over Time')
        ax2.fill_between(time_steps, rev_cumulative, color='green', alpha=0.3)  # Area sottesa con alpha 0.3
        ax2.set_ylabel('Cumulative Revenues [Euros]')
        ax2.legend(loc='upper left')

        # Nuovo grafico per rev_pv e rev_bess (quarto grafico) in basso a destra
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
        if minimize_C:
            plt.savefig(os.path.join(self.plots_dir, "POD_View_minC.png"))
        else:
            plt.savefig(os.path.join(self.plots_dir, "POD_View.png"))


    def Total_View(self, num_values):
        time_steps = self.time_steps[:num_values]
        charged_energy = self.charged_energy[:num_values]
        discharged_energy = self.discharged_energy[:num_values]
        pun_values = self.PUN_timeseries[:num_values]
        soc = self.soc[:num_values]
        produced_from_pv = self.produced_from_pv[:num_values]
        taken_from_pv = self.taken_from_pv[:num_values]  # Energia da PV
        taken_from_grid = self.taken_from_grid[:num_values]  # Energia dalla rete
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
        total_dicharged_pv = np.sum(-discharged_from_pv)

        curtailment = np.maximum(produced_from_pv + discharged_from_pv - taken_from_pv, 0.0)
        total_curtailment = np.sum(curtailment)

        # Creazione del layout con 4 box usando gridspec
        fig = plt.figure(figsize=(24, 12))  # Aumentato il figsize per adattarsi a 4 grafici
        gs = gridspec.GridSpec(2, 2, width_ratios=[1, 1], height_ratios=[1, 1])

        # Asse per SoC in alto a sinistra
        ax0 = fig.add_subplot(gs[0, 0])

        # Normalize SoC values to be in the range [0, 1] for the colormap
        norm = Normalize(vmin=min(soc*100), vmax=max(soc*100))
        cmap = matplotlib.colors.LinearSegmentedColormap.from_list("", ["lightblue", "steelblue", "darkblue"])

        # Plot SoC with gradient colored bars based on value
        for i in range(len(time_steps)):
            ax0.bar(time_steps[i], soc[i]*100, color=cmap(norm(soc[i])))
        ax0.set_title('State of Charge (SoC)')
        ax0.set_ylabel('SoC [%]')

        # Asse per l'energia caricata e scaricata (secondo grafico) in basso a sinistra
        ax1 = fig.add_subplot(gs[1, 0])

        # Aggiungi l'area sottesa per 'produced_from_pv' in un giallo più acceso e dietro le barre
        ax1.fill_between(time_steps, 0, produced_from_pv, color='lightblue', alpha=0.3, label='Produced from PV')

        width = 0.4
        ax1.bar(time_steps, [1] * np.array(taken_from_grid), width=width, color='darkgreen',
                label='From Grid to BESS')
        ax1.bar(time_steps, taken_from_pv, color='darkblue', bottom=-discharged_from_pv, width= width,
                label='From PV to BESS')

        ax1.bar(time_steps, discharged_energy, width=width, color='darkred',
               bottom= np.array(taken_from_grid),
               label='From BESS to Grid')
        ax1.bar(time_steps, -discharged_from_pv, width= width,
                label='From PV to Grid',)

        ax1.set_ylabel('Energy [kWh]')
        ax1.set_title('System Energy Flows')
        ax1.legend(loc='upper left')
        plt.ylim(-size * 0.6, size * 0.6)

        # Plot PUN values on the secondary axis
        ax3 = ax1.twinx()
        ax3.plot(time_steps, pun_values, color='black', label='PUN')
        ax3.set_ylabel('PUN [Euro/MWh]')
        ax3.legend(loc='upper right')

        ax1.set_xlabel('Time Window [h]')

        # Asse per la cumulata dei ricavi a destra del primo grafico
        ax2 = fig.add_subplot(gs[0, 1])  # Usa solo la prima riga nella colonna di destra
        ax2.plot(time_steps, rev_cumulative, color='lightgreen', label='Cumulative Revenues', alpha=1)
        ax2.set_title('Cumulative Revenues Over Time')
        ax2.fill_between(time_steps, rev_cumulative, color='green', alpha=0.3)  # Area sottesa con alpha 0.3
        ax2.set_ylabel('Cumulative Revenues [Euros]')
        ax2.legend(loc='upper left')

        # Nuovo grafico per rev_pv e rev_bess (quarto grafico) in basso a destra
        colors_bess = ['red' if total < 0 else 'limegreen' for total in rev_pv]

        ax4 = fig.add_subplot(gs[1, 1])


        if total_curtailment != 0.0:
            def func(pct, allvalues):
                absolute = int(np.round(pct / 100. * np.sum(allvalues)))  # Calculate absolute value
                return f"{absolute} ({pct:.1f}%)"  # Format as "absolute (percentage)"

            sizes = [total_curtailment, total_dicharged_pv, total_from_pv, total_from_grid, total_discharged]
            labels = ['Curtailment', 'From PV to Grid', 'From PV to BESS', 'From Grid to BESS', 'From BESS to Grid']
            colors = ['Orange', 'khaki', 'steelblue', 'lightblue', 'paleturquoise']
            explode = (0.05, 0.05,0.05,0.05,0.05)  # explode the first slice (Total Curtailment)

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

            sizes = [total_dicharged_pv, total_from_pv, total_from_grid, total_discharged]
            labels = [ 'From PV to Grid', 'From PV to BESS', 'From Grid to BESS', 'From BESS to Grid']
            colors = [ 'goldenrod', 'steelblue', 'lightblue', 'paleturquoise']
            explode = ( 0.05, 0.05, 0.05, 0.05)  # explode the first slice (Total Curtailment)

            # Create pie chart
            ax4.pie(sizes, explode=explode, labels=labels, colors=colors,
                    autopct=lambda pct: func(pct, sizes),  # Use the custom function
                    shadow=False, startangle=90, textprops={'fontsize': 14})
            ax4.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
            ax4.set_title('System Total Energy Distribution [kWh]')


        fig.tight_layout()

        # Save the plot based on the minimize_C condition
        if minimize_C:
            plt.savefig(os.path.join(self.plots_dir, "Total_View_minC.png"))
        else:
            plt.savefig(os.path.join(self.plots_dir, "Total_View.png"))

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

        plt.savefig("Plots/charge_rate.png")
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
        plt.savefig("Plots/charge_rate.png")
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

        plt.savefig("Plots/disc_rate.png")


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

        if minimize_C:
            plt.savefig('Plots/minimize_C_rate/total_convergence_minC.png')
        else:
            plt.savefig('Plots/optimal_C_rate/total_convergence.png')

    @staticmethod
    def PUN_plot(PUN_timeseries):

        # Line plot dei valori PUN (terza colonna del DataFrame)

        pun_values = PUN_timeseries  # Estrazione della terza colonna (indice 2)
        plt.figure(figsize=(12, 8))
        plt.plot(pun_values, marker='o', color='b')
        plt.title('PUN Values')
        plt.xlabel('Time step')
        plt.ylabel('PUN Value')
        plt.savefig(os.path.join("Plots", "PUN.png"))
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

            if minimize_C:
                output_dir = 'Plots/minimize_C_rate'
            else:
                output_dir = 'Plots/optimal_C_rate'

            if not os.path.exists(output_dir):
                os.makedirs(output_dir)

            # Save the figure in the "Plots" folder

            if minimize_C:
                output_path = os.path.join(output_dir, f'convergence_{fig_num + 1}_minC.png')
                fig.savefig(output_path)
            else:
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

        if minimize_C:
            output_dir = 'Plots/minimize_C_rate'
        else:
            output_dir = 'Plots/optimal_C_rate'

        os.makedirs(output_dir, exist_ok=True)
        if minimize_C:
            output_path = os.path.join(output_dir, 'C_rate_PUN_minC.png')
        else:
            output_path = os.path.join(output_dir, 'C_rate_PUN.png')

        plt.savefig(output_path)





