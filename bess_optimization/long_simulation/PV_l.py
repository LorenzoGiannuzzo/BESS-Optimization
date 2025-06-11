""" BESS Optimization using NSGA-III Algorithm

    __author__ = "Lorenzo Giannuzzo"
    __maintainer__ = "Lorenzo Giannuzzo"
    __email__ = "lorenzo.giannuzzo@polito.it"
    __status__ = "in progress"
    __version__ = "v0.2.1"
    __license__ = "MIT"

Last Update of current code: 05/06/2025 """

import pandas as pd
from argparser_l import input_PV_path, PV_power, rec_production
from Economic_parameters_l import time_window
import numpy as np
from logger import setup_logger

a = 0
b = 0

if input_PV_path == 0.0:

    columns = ["time", "P"]

    # Create a DataFrame filled with zeros for all columns except 'time'
    # For 'time', create strings of zeros of same length as example timestamps (13 chars)
    time_zeros = [0.0] * time_window

    pv_production = pd.DataFrame({
        'time': time_zeros,
        'P': np.zeros(time_window)
    })

    a = 1

if rec_production == 0.0:

    time_zeros = [0.0] * time_window

    rec_pv = pd.DataFrame({
        'time': time_zeros,
        'P': np.zeros(time_window)
    })

    rec_pv = rec_pv.to_numpy()

    b = 1

# File path
file_path = input_PV_path

if a == 0:

    # Read the PV production data
    pv_production = pd.read_csv(file_path, sep=';', usecols=['time', 'P'])

    # Scale the power values
    pv_production['P'] = pv_production['P'] * PV_power / 1000


# REC PRODUCTION -------------------------------------------------------------------------------------------------------

if b == 0:

    # File path
    file_path_rec = rec_production

    # Read the PV production data
    rec_production = pd.read_csv(file_path_rec, sep=';', usecols=['time', 'P'])

    rec_production['P'] = pd.to_numeric(rec_production['P'], errors='coerce')

    # Scale the power values
    rec_production['P'] = rec_production['P'] * PV_power / 1000

    rec_pv = rec_production.to_numpy()

