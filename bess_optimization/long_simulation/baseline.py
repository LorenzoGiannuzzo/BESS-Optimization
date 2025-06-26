# TODO: the path file is hardcoded, should be a new argparser input

import pandas as pd
import numpy as np

# Specifica il percorso del file CSV
file_path = r"C:\Users\lorenzo.giannuzzo\PycharmProjects\BESS-Optimization\data\Input\baseline\baseline.csv"

# Legge il CSV usando pandas
try:
    df = pd.read_csv(file_path)
    #print("File caricato con successo!")
    #print(df.head())  # Mostra le prime 5 righe
except FileNotFoundError:
    print(f"Il file non è stato trovato al percorso: {file_path}")
except Exception as e:
    print(f"Errore durante la lettura del file: {e}")

baseline_data = df
