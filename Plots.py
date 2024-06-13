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
        plt.bar(self.time_steps, self.charged_energy, color='g')
        plt.title('Charged Energy')
        plt.xlabel('Time Window [h]')
        plt.ylabel('Charged Energy [kWh]')
        plt.savefig(os.path.join(self.plots_dir, "Charged_Energy.png"))
        plt.close()

    def plot_discharged_energy(self):
        plt.figure(figsize=(12, 8))
        plt.bar(self.time_steps, self.discharged_energy, color='r')
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
        cmap = plt.get_cmap('RdYlGn')

        # Plot SoC with gradient colored bars based on value
        for i in range(len(time_steps_24)):
            ax0.bar(time_steps_24[i], soc_24[i], color=cmap(norm(soc_24[i])))

        ax0.set_title('State of Charge (SoC) [%]')
        ax0.set_ylabel('SoC')

        # Plot charged and discharged energy with PUN on the second subplot
        width = 0.4
        ax1.bar(time_steps_24 - width / 2, charged_energy_24, width=width, color='g', label='Charged Energy [kWh]')
        ax1.bar(time_steps_24 + width / 2, discharged_energy_24, width=width, color='r',
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
