from utils import Get_data
from BESS_parameters import file_path, file_path2, sheetname, sheetname2, sheetname3
import matplotlib.pyplot as plt
import os


# DEFINE OPTIMIZATION TIME WINDOW

time_window_e = 48

# IMPORT PUN TIMESERIES

data = Get_data.get_data(file_path2, sheetname3)
PUN_timeseries = data.iloc[:time_window_e, 2].to_numpy()

import matplotlib
matplotlib.use('Agg')

# Line plot dei valori PUN (terza colonna del DataFrame)
pun_values = PUN_timeseries  # Estrazione della terza colonna (indice 2)
plt.figure(figsize=(12, 8))
plt.plot(pun_values, marker='o', color='b')
plt.title('PUN Values')
plt.xlabel('Time step')
plt.ylabel('PUN Value')
plt.savefig(os.path.join("Plots", "PUN_values_plot.png"))
plt.close()
