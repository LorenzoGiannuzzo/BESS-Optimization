""" BESS Optimization using NSGA-III Algorithm

    __author__ = "Lorenzo Giannuzzo"
    __maintainer__ = "Lorenzo Giannuzzo"
    __email__ = "lorenzo.giannuzzo@polito.it"
    __status__ = "in progress"
    __version__ = "v0.2.1"
    __license__ = "MIT"

Last Update of current code: 05/06/2025 """

import pandas as pd
from argparser_s import input_PV_path, PV_power, rec_production
from Economic_parameters_s import time_window
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

    # Convert 'time' to pandas datetime object
    # The format is 'YYYYMMDD:HHMM', so we need to replace ':' with a space and parse it
    pv_production['time'] = pd.to_datetime(pv_production['time'].str.replace(':', ' '), format='%Y%m%d %H%M', utc=True)

    # Set 'time' as the index
    pv_production.set_index('time', inplace=True)

    # Create new columns for month and hour
    pv_production['month'] = pv_production.index.month
    pv_production['hour'] = pv_production.index.hour

    # Group by month and hour, then calculate the mean
    mean_hourly_pv = pv_production.groupby(['month', 'hour'])['P'].mean().reset_index()

    # Create a new DataFrame to hold the final result
    final_pv_result = pd.DataFrame()

    # Iterate over each month to create the final structure
    for month in range(1, 13):  # Months from 1 to 12
        month_data = mean_hourly_pv[mean_hourly_pv['month'] == month]

        # Append the month data to the final result
        final_pv_result = pd.concat([final_pv_result, month_data[['hour', 'P']]], ignore_index=True)

    # Reset index to get hours as a column
    final_pv_result.reset_index(drop=True, inplace=True)

    # Convert to numpy array
    PV_timeseries_reduced = final_pv_result
    pv_production = PV_timeseries_reduced

# REC PRODUCTION -------------------------------------------------------------------------------------------------------

if b == 0:

    # File path
    file_path_rec = rec_production

    # Read the PV production data
    rec_production = pd.read_csv(file_path_rec, sep=';', usecols=['time', 'P'])

    rec_production['P'] = pd.to_numeric(rec_production['P'], errors='coerce')

    # Scale the power values
    rec_production['P'] = rec_production['P'] * PV_power / 1000

    # Convert 'time' to pandas datetime object
    # The format is 'YYYYMMDD:HHMM', so we need to replace ':' with a space and parse it
    rec_production['time'] = pd.to_datetime(rec_production['time'].str.replace(':', ' '), format='%Y%m%d %H%M', utc=True)

    # Set 'time' as the index
    rec_production.set_index('time', inplace=True)

    # Create new columns for month and hour
    rec_production['month'] = rec_production.index.month
    rec_production['hour'] = rec_production.index.hour

    # Group by month and hour, then calculate the mean
    mean_hourly_pv = rec_production.groupby(['month', 'hour'])['P'].mean().reset_index()

    # Create a new DataFrame to hold the final result
    final_pv_result_rec = pd.DataFrame()

    # Iterate over each month to create the final structure
    for month in range(1, 13):  # Months from 1 to 12
        month_data = mean_hourly_pv[mean_hourly_pv['month'] == month]

        # Append the month data to the final result
        final_pv_result_rec = pd.concat([final_pv_result_rec, month_data[['hour', 'P']]], ignore_index=True)

    # Reset index to get hours as a column
    final_pv_result_rec.reset_index(drop=True, inplace=True)

    # Convert to numpy array
    PV_timeseries_reduced = final_pv_result_rec
    rec_pv = PV_timeseries_reduced.to_numpy()
