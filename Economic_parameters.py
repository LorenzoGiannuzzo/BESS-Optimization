"""

BESS Optimization using NSGA-III Algorithm

    __author__ = "Lorenzo Giannuzzo"
    __maintainer__ = "Lorenzo Giannuzzo"
    __email__ = "lorenzo.giannuzzo@polito.it"
    __status__ = "in progress"
    __version__ = "v0.2.1"
    __license__ = "MIT"

Last Update of current code: 06/11/2024 - 17:36

"""

# IMPORT LIBRARIES

import pandas as pd
from argparser import input_json_path

# SET PATH FILE

json_file_path = input_json_path

df = pd.read_json(json_file_path)
df['value'] = df['value'] / 1000000

PUN_timeseries = df
PUN_timeseries = PUN_timeseries.to_numpy()

# EXTRACTIN OPTIMIZAITON TIME WINDOW

time_window = len(PUN_timeseries[:,1])

#tst

