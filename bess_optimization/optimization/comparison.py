import matplotlib.pyplot as plt
import numpy as np
import os

# Dati originali
data = {
    'AUTUMN': [3.01, 1.66, 1.67, 1.61, 2.92, 0.91],
    'SPRING': [5.15, 4.6, 4.5, 4.4, 4.18, 3.68],
    'SUMMER': [4.46, 3.62, 3.84, 3.75, 3.74, 2.34],
    'WINTER': [3.8, 2.64, 3.09, 3.25, 2.98, 2.46]
}

algorithms = ['Benchmark', 'NSGA2', 'NSGA3', 'RNSGA3', 'SPEA2', 'BRKGA']
colors_dict = {
    'Benchmark': '#1f77b4',
    'NSGA2': '#ff7f0e',
    'NSGA3': '#2ca02c',
    'RNSGA3': '#d62728',
    'SPEA2': '#9467bd',
    'BRKGA': '#8c564b'
}

# Crea i sottoplot
fig, axs = plt.subplots(2, 2, figsize=(14, 10))
axs = axs.flatten()

for i, (season, values) in enumerate(data.items()):
    normalized = (np.array(values) / values[0]) * 100  # Converti in percentuale

    # Ordina decrescente
    alg_and_vals = list(zip(algorithms, normalized))
    sorted_data = sorted(alg_and_vals, key=lambda x: x[1], reverse=True)
    sorted_algorithms, sorted_values = zip(*sorted_data)
    sorted_colors = [colors_dict[alg] for alg in sorted_algorithms]

    axs[i].bar(sorted_algorithms, sorted_values, color=sorted_colors)
    axs[i].set_title(f'{season} - Normalized Revenues Relative Error [%]')
    axs[i].set_ylabel('Relative Error [%]')
    axs[i].set_ylim(0, 110)
    axs[i].tick_params(axis='x', rotation=45)

plt.tight_layout()

# Salva la figura
output_path = r'C:\Users\lorenzo.giannuzzo\PycharmProjects\BESS-Optimization\data\output\normalized_relative_error_percent.pdf'
plt.savefig(output_path)