import pandas as pd
import numpy as np
import json
import os

def analyze_load_profile(file_path, sheet_name='Sheet 1', output_path='../../data/Output/load_description/user_statistics.json'):
    # Carica il file Excel
    df = pd.read_excel(file_path, sheet_name=sheet_name)

    # Controlla che le colonne necessarie siano presenti
    if 'Data' not in df.columns or 'value' not in df.columns:
        raise ValueError("Il file Excel deve contenere almeno le colonne 'Data' e 'value'")

    # Converti la colonna 'Data' in formato datetime
    df['Data'] = pd.to_datetime(df['Data'], errors='coerce')

    # Assicurati che 'value' sia numerico
    df['value'] = pd.to_numeric(df['value'], errors='coerce')

    # Elimina righe non valide
    df.dropna(subset=['Data', 'value'], inplace=True)

    # Imposta 'Data' come indice
    df.set_index('Data', inplace=True)

    # Raggruppamento mensile per energia (in kWh) - assumendo dati orari in kW
    monthly_energy = df['value'].resample('M').sum()
    annual_energy = monthly_energy.sum()

    # Calcolo statistiche
    stats = {
        "peak_load_kW": df['value'].max(),
        "mean_load_kW": df['value'].mean(),
        "annual_energy_consumption_kWh": annual_energy,
        "average_monthly_energy_consumption_kWh": monthly_energy.mean(),
        "minimum_monthly_energy_consumption_kWh": monthly_energy.min(),
        "maximum_monthly_energy_consumption_kWh": monthly_energy.max(),
        "maximum_hourly_energy_consumption_kW": df['value'].max()
    }

    # Crea la cartella se non esiste
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Salva il file JSON
    with open(output_path, 'w') as f:
        json.dump(stats, f, indent=4)

    print(f"[INFO] Statistiche salvate in {output_path}")

# === ESECUZIONE ===

analyze_load_profile("../../data/Loads/BTA6_5.XLSX")