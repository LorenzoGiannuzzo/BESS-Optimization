"""

BESS Optimization using NSGA-III Algorithm

    __author__ = "Lorenzo Giannuzzo"
    __maintainer__ = "Lorenzo Giannuzzo"
    __email__ = "lorenzo.giannuzzo@polito.it"
    __status__ = "in progress"
    __version__ = "v0.2.1"
    __license__ = "MIT"

Last Update of current code: 09/01/2025 - 17:38

"""

# IMPORT LIBRARIES

import configuration_s
import os
import pandas as pd
import json

from argparser_s import input_load
from pymoo.optimize import minimize
from objective_function_s import Revenues
from configuration_s import plot

def analyze_user_profile(rec_load_path, sheet_name='Sheet 1', output_folder='data/Output/Short_Simulation'):

    # Carica file Excel
    df_rec = pd.read_excel(rec_load_path, sheet_name=sheet_name)

    # Converti 'Data' in datetime
    df_rec['Data'] = pd.to_datetime(df_rec['Data'], errors='coerce')

    # Assicurati che 'value' sia numerico
    df_rec['value'] = pd.to_numeric(df_rec['value'], errors='coerce')
    df_rec.dropna(subset=['Data', 'value'], inplace=True)

    # Imposta 'Data' come indice
    df_rec.set_index('Data', inplace=True)

    # Energia mensile in kWh (supponendo dati orari in kW)
    monthly_energy = df_rec['value'].resample('M').sum()
    annual_energy = monthly_energy.sum()

    # Calcola statistiche
    stats = {
        "peak_load_kW": df_rec['value'].max(),
        "mean_load_kW": df_rec['value'].mean(),
        "annual_energy_consumption_kWh": annual_energy,
        "average_monthly_energy_consumption_kWh": monthly_energy.mean(),
        "minimum_monthly_energy_consumption_kWh": monthly_energy.min(),
        "maximum_monthly_energy_consumption_kWh": monthly_energy.max(),
        "maximum_hourly_energy_consumption_kW": df_rec['value'].max()
    }

    # Crea directory se non esiste
    os.makedirs(output_folder, exist_ok=True)

    # Salva il file JSON
    output_path = os.path.join(output_folder, 'user_statistics.json')
    with open(output_path, 'w') as f:
        json.dump(stats, f, indent=4)

    print(f"[INFO] User statistics saved to {output_path}")

