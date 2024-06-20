import pandas as pd

# IMPORT PUN TIMESERIES

json_file_path = r'C:\Users\lorenzo.giannuzzo\PycharmProjects\BESS-Optimization\pun.json'
df = pd.read_json(json_file_path)
df['value'] = df['value'] / 1000000

PUN_timeseries = df

PUN_timeseries = PUN_timeseries.to_numpy()
time_window = len(PUN_timeseries[:,1])

