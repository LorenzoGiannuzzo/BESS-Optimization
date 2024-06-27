import pandas as pd
from argparser import input_json_path


# IMPORT PUN TIMESERIES

json_file_path = input_json_path

df = pd.read_json(json_file_path)

df['value'] = df['value'] / 1000000

PUN_timeseries = df

PUN_timeseries = PUN_timeseries.to_numpy()

time_window = len(PUN_timeseries[:,1])

