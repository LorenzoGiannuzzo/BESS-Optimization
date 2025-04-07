import pandas as pd
import numpy as np
import os
import ExcelOpener_l
import matplotlib.colors as mcolors
import matplotlib.gridspec as gridspec
from matplotlib.colors import Normalize
import matplotlib


def euclidean_distance_percentage_exclude_first(df1, df2):
    # Ensure both DataFrames have the same shape
    if df1.shape != df2.shape:
        raise ValueError("DataFrames must have the same shape")


    # Exclude the first column
    df1_relevant = df1.iloc[:, 1:]  # Exclude the first column
    df2_relevant = df2.iloc[:, 1:]  # Exclude the first column

    df1_relevant = df1_relevant.drop(df1_relevant.columns[9], axis=1)
    df2_relevant = df2_relevant.drop(df2_relevant.columns[9], axis=1)

    df1_relevant = df1_relevant.drop(df1_relevant.columns[10], axis=1)
    df2_relevant = df2_relevant.drop(df2_relevant.columns[10], axis=1)

    col_names = df1_relevant.iloc[0,:].index

    df1_relevant = np.array(df1_relevant, dtype=float)
    df2_relevant = np.array(df2_relevant,dtype=float)

    num_columns = df1_relevant.shape[1]
    num_rows = df1_relevant.shape[0]

    distances_matrix = np.zeros((num_rows, num_columns))
    # Calcoliamo la distanza euclidea percentuale per ogni coppia di colonne
    for i in range(num_columns):
        for j in range(num_rows):

            # Calcoliamo la distanza euclidea tra la colonna i di df1 e la colonna j di df2
            distance = df1_relevant[j, i] - df2_relevant[j, i]

            distances_matrix[j,i] = distance

    distances_matrix = pd.DataFrame(distances_matrix, columns = col_names)

    # Restituisce la matrice delle distanze
    return distances_matrix

# ----------------------------------------------------------------------------------------------------------------------

def plot_comparison(distance):

    distance = np.array(distance)

    norm = Normalize(vmin=-10.0, vmax=15.0)
    cmap = matplotlib.colors.LinearSegmentedColormap.from_list("", ["orange", "darkorchid", "indigo"])

    pv_to_bess = np.abs(distance[:,11])
    grid_to_bess = np.abs(distance[:,12])
    pv_to_grid = np.abs(distance[:,13])
    bess_to_grid = -np.abs(distance[:,14])
    bess_to_load = np.abs(distance[:,15])
    pv_to_load = np.abs(distance[:,16])
    time_steps = np.arange(24)

    # PLOTTING DISCHARGED ENERGY, TAKEN FROM PV, AND TAKEN FROM GRID
    bar_width = 0.35  # Adjusted bar width

    import matplotlib.pyplot as plt

    plt.figure(figsize=(14, 14))  # Increased figure size for better visibility

    # Create primary axis for energy flows
    ax1 = plt.subplot(211)  # First subplot for energy flows

    # Bar plot for discharged energy
    bar1 = ax1.bar(time_steps, bess_to_grid, bar_width, label='BESS to Grid', color='darkred')


    # Bar plot for taken from PV (lighter green)
    bar3 = ax1.bar(time_steps, pv_to_bess, bar_width, label='PV to BESS', color='lightgreen', alpha=0.8,
                       bottom=pv_to_load)

    # Bar plot for taken from grid (darker green) - now paired with other bars
    bar2 = ax1.bar(time_steps + bar_width, grid_to_bess, bar_width, label='Grid to BESS', color='darkgreen',
                       alpha=0.8)

    # Bar plot for from PV to load
    bar4 = ax1.bar(time_steps, pv_to_load, bar_width, label='PV to Load', color='indianred', alpha=0.8)

    # Bar plot for from BESS to load, starting from the top of the discharged energy bar
    bar5 = ax1.bar(time_steps, bess_to_load, bar_width, label='BESS to Load', color='steelblue', alpha=0.8,
                       bottom=np.abs(pv_to_load) + np.abs(pv_to_bess))

    bar6 = ax1.bar(time_steps, -pv_to_grid, bar_width, label='PV to Grid', color='aquamarine', alpha=0.8,
                       bottom=pv_to_bess + pv_to_load)

    # Set labels and title
    ax1.set_xlabel('Hours', fontsize=16)  # Increased font size for better visibility
    ax1.set_ylabel('Energy [kWh]', fontsize=16)  # Increased font size for better visibility
    plt.title('Energy Flows', fontsize=16)  # Increased font size for better visibility
    ax1.set_xticks(time_steps + bar_width / 2)
    ax1.set_xticklabels(time_steps, rotation=0)

    # Legend for bar plots
    bar_handles = [bar1, bar2, bar3, bar4, bar5, bar6]
    bar_labels = ['BESS to Grid', 'Grid to BESS', 'PV to BESS', 'PV to Load',
                      'BESS to Load', 'PV to Grid']


    max_energy = 15
    min_energy = -7.5
    ax1.set_ylim(min_energy * 1.1, max_energy * 1.1)

    # Add horizontal dashed gray grid lines
    ax1.yaxis.grid(True, linestyle='--', color='gray', alpha=0.3)  # Adjust alpha for visibility

    plt.tight_layout()

        # Save the plot
    plt.savefig(os.path.join(r'C:\Users\lorenzo.giannuzzo\PycharmProjects\BESS-Optimization\data\output\xlsx\nsga3', "View Comparison.pdf"), dpi=500)
    plt.close()


# ----------------------------------------------------------------------------------------------------------------------

path_benchmark_autumn = r'C:\Users\lorenzo.giannuzzo\PycharmProjects\BESS-Optimization\data\output\xlsx\results_autumn_benchmark.xlsx'
path_benchmark_spring = r'C:\Users\lorenzo.giannuzzo\PycharmProjects\BESS-Optimization\data\output\xlsx\results_spring_benchmark.xlsx'
path_benchmark_summer = r'C:\Users\lorenzo.giannuzzo\PycharmProjects\BESS-Optimization\data\output\xlsx\results_summer_benchmark.xlsx'
path_benchmark_winter = r'C:\Users\lorenzo.giannuzzo\PycharmProjects\BESS-Optimization\data\output\xlsx\result_winter_benchmark.xlsx'

benchmark_autumn = pd.read_excel(path_benchmark_autumn, 'Sheet1')
benchmark_spring = pd.read_excel(path_benchmark_spring, 'Sheet1')
benchmark_summer = pd.read_excel(path_benchmark_summer,'Sheet1')
benchmark_winter = pd.read_excel(path_benchmark_winter,'Sheet1')

path_nsga3_autumn = r'C:\Users\lorenzo.giannuzzo\PycharmProjects\BESS-Optimization\data\output\xlsx\nsga3\results_autumn_nsga3.xlsx'
path_nsga3_spring = r'C:\Users\lorenzo.giannuzzo\PycharmProjects\BESS-Optimization\data\output\xlsx\nsga3\results_spring_nsga3.xlsx'
path_nsga3_summer = r'C:\Users\lorenzo.giannuzzo\PycharmProjects\BESS-Optimization\data\output\xlsx\nsga3\results_summer_nsga3.xlsx'
path_nsga3_winter = r'C:\Users\lorenzo.giannuzzo\PycharmProjects\BESS-Optimization\data\output\xlsx\nsga3\results_winter_nsga3.xlsx'

nsga3_autumn = pd.read_excel(path_nsga3_autumn, 'Sheet1')
nsga3_spring = pd.read_excel(path_nsga3_spring, 'Sheet1')
nsga3_summer = pd.read_excel(path_nsga3_summer, 'Sheet1')
nsga3_winter = pd.read_excel(path_nsga3_winter, 'Sheet1')

percentage_distances = euclidean_distance_percentage_exclude_first(benchmark_autumn, nsga3_autumn)
percentage_distances = pd.DataFrame(percentage_distances)

excel_file_path = r'C:\Users\lorenzo.giannuzzo\PycharmProjects\BESS-Optimization\data\output\xlsx\nsga3\comparison.xlsx'  # Define your output path for the Excel file
percentage_distances.to_excel(excel_file_path, index=False)

plot_comparison(percentage_distances)


