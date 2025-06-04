import pandas as pd
import numpy as np
from argparser_l import input_PV
from Economic_parameters_l import time_window

# File path
file_path = input_PV

if input_PV == 0.0:

    columns = ["time", "P", "G(i)", "H_sun", "T2m", "WS10m", "Int"]

    # Create a DataFrame filled with zeros for all columns except 'time'
    # For 'time', create strings of zeros of same length as example timestamps (13 chars)
    time_zeros = [0.0] * time_window

    # For other columns, fill with numeric zeros
    pv_production = {
        "time": time_zeros,
        "P": np.zeros(time_window),
        "G(i)": np.zeros(time_window),
        "H_sun": np.zeros(time_window),
        "T2m": np.zeros(time_window),
        "WS10m": np.zeros(time_window),
        "Int": np.zeros(time_window)
    }

else:

    pv_production = pd.read_csv(file_path, sep=';')
    #pv_production = pd.read_csv(file_path, sep=';')

    from argparser_l import PV_power

    #pv_production = pv_production[pv_production['Season'] == season].reset_index()
    pv_production['P'] = pv_production['P'] * PV_power / 1000


