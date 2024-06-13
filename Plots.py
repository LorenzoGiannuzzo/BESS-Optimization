import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib

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
        plt.bar(self.time_steps, self.soc, color='lightblue')
        plt.title('State of Charge (SoC) [%]')
        plt.xlabel('Time step')
        plt.ylabel('SoC')
        plt.savefig(os.path.join(self.plots_dir, "SoC_hist.png"))
        plt.close()

    def plot_charged_energy(self):
        plt.figure(figsize=(12, 8))
        plt.bar(self.time_steps, self.charged_energy, color='g')
        plt.title('Charged Energy')
        plt.xlabel('Time step')
        plt.ylabel('Charged Energy')
        plt.savefig(os.path.join(self.plots_dir, "Charged_Energy_hist.png"))
        plt.close()

    def plot_discharged_energy(self):
        plt.figure(figsize=(12, 8))
        plt.bar(self.time_steps, self.discharged_energy, color='r')
        plt.title('Discharged Energy')
        plt.xlabel('Time step')
        plt.ylabel('Discharged Energy')
        plt.savefig(os.path.join(self.plots_dir, "Discharged_Energy_hist.png"))
        plt.close()

    def plot_combined_energy_with_pun(self, num_values):
        time_steps_24 = self.time_steps[:num_values]
        charged_energy_24 = self.charged_energy[:num_values]
        discharged_energy_24 = self.discharged_energy[:num_values]
        pun_values_24 = self.PUN_timeseries[:num_values]

        fig, ax1 = plt.subplots(figsize=(12, 8))
        width = 0.4
        ax1.bar(time_steps_24 - width / 2, charged_energy_24, width=width, color='g', label='Charged Energy [kWh]')
        ax1.bar(time_steps_24 + width / 2, discharged_energy_24, width=width, color='r', label='Discharged Energy [kWh]')
        ax1.set_xlabel('Time step')
        ax1.set_ylabel('Energy [kWh]')
        ax1.set_title('Charged and Discharged Energy with PUN')
        ax1.legend(loc='upper left')

        ax2 = ax1.twinx()
        ax2.plot(time_steps_24, pun_values_24, color='black', label='PUN [Euro/MWh]')
        ax2.set_ylabel('PUN Value')
        ax2.legend(loc='upper right')

        fig.tight_layout()
        plt.savefig(os.path.join(self.plots_dir, "Charged_and_Discharged_Energy_with_PUN.png"))
        plt.close()