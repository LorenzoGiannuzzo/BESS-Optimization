""" BESS Optimization using NSGA-III Algorithm

    __author__ = "Lorenzo Giannuzzo"
    __maintainer__ = "Lorenzo Giannuzzo"
    __email__ = "lorenzo.giannuzzo@polito.it"
    __status__ = "in progress"
    __version__ = "v0.2.1"
    __license__ = "MIT"

Last Update of current code: 21/02/2025 """

# IMPORT LIBRARIES AND MODULES -----------------------------------------------------------------------------------------

import pandas as pd
from argparser_s import input_json_path
from logger import setup_logger

# GET LOGGER
setup_logger()

# SET PATH FILE
json_file_path = input_json_path

# Read the JSON file into a DataFrame
df = pd.read_json(json_file_path)

# Convert 'datetime' to pandas datetime object, handling the 'Z' at the end
df['datetime'] = pd.to_datetime(df['datetime'].str.replace('Z', ''), utc=True)

# Set 'datetime' as the index
df.set_index('datetime', inplace=True)

# Convert 'value' to numeric and divide by 1,000,000
df['value'] = pd.to_numeric(df['value'], errors='coerce') / 1000000

# Drop rows with NaN values in 'value'
df.dropna(subset=['value'], inplace=True)

# Create a DataFrame for weekdays (Monday to Friday)
df_weekdays = df[~df.index.weekday.isin([5, 6])]  # Exclude Saturdays (5) and Sundays (6)

df_weekdays = df[df.index.weekday.isin([0,1,2,3,4,5,6])]  # Exclude Saturdays (5) and Sundays (6)

# Create a DataFrame for weekends (Saturday and Sunday)
df_weekends = df[df.index.weekday.isin([5, 6])]  # Include only Saturdays (5) and Sundays (6)

# Create a new column for the month and hour in the weekdays DataFrame
df_weekdays['month'] = df_weekdays.index.month
df_weekdays['hour'] = df_weekdays.index.hour

# Group by month and hour for weekdays, then calculate the mean
mean_hourly_weekdays = df_weekdays.groupby(['month', 'hour'])['value'].mean().reset_index()

# Create a new DataFrame for the final result of weekdays
final_result_weekdays = pd.DataFrame()

# Iterate over each month to create the final structure for weekdays
for month in range(1, 13):  # Months from 1 to 12
    month_data = mean_hourly_weekdays[mean_hourly_weekdays['month'] == month]
    # Append the month data to the final result for weekdays
    final_result_weekdays = pd.concat([final_result_weekdays, month_data[['hour', 'value']]], ignore_index=True)

# Reset index to get hours as a column for weekdays
final_result_weekdays.reset_index(drop=True, inplace=True)

# Filter the DataFrame to include only the 15th of each month
df_15th = df[df.index.day == 15]

# Create a new column for the month and hour in the 15th DataFrame
df_15th['month'] = df_15th.index.month
df_15th['hour'] = df_15th.index.hour

# Group by month and hour for the 15th, then calculate the mean
mean_hourly_15th = df_15th.groupby(['month', 'hour'])['value'].mean().reset_index()

# Create a new DataFrame for the final result of the 15th
final_result_15th = pd.DataFrame()

# Iterate over each month to create the final structure for the 15th
for month in range(1, 13):  # Months from 1 to 12
    month_data = mean_hourly_15th[mean_hourly_15th['month'] == month]
    # Append the month data to the final result for the 15th
    final_result_15th = pd.concat([final_result_15th, month_data[['hour', 'value']]], ignore_index=True)

# Reset index to get hours as a column for the 15th
final_result_15th.reset_index(drop=True, inplace=True)
# TODO: FORCING THE EVALUATION TO THE 15th OF EACH MONTH
final_result_weekdays = final_result_15th

# Create a new column for the month and hour in the weekends DataFrame
df_weekends['month'] = df_weekends.index.month
df_weekends['hour'] = df_weekends.index.hour

# Group by month and hour for weekends, then calculate the mean
mean_hourly_weekends = df_weekends.groupby(['month', 'hour'])['value'].mean().reset_index()

# Create a new DataFrame for the final result of weekends
final_result_weekends = pd.DataFrame()

# Iterate over each month to create the final structure for weekends
for month in range(1, 13):  # Months from 1 to 12
    month_data = mean_hourly_weekends[mean_hourly_weekends['month'] == month]
    # Append the month data to the final result for weekends
    final_result_weekends = pd.concat([final_result_weekends, month_data[['hour', 'value']]], ignore_index=True)

# Reset index to get hours as a column for weekends
final_result_weekends.reset_index(drop=True, inplace=True)

from argparser_s import weekends

if weekends == 'True':
    # Convert to numpy arrays if needed
    PUN_timeseries = final_result_weekdays.to_numpy()

else:
    PUN_timeseries = final_result_weekends.to_numpy()

# TODO BRUTE FORCE PUN TO BE THE ONE EVALUATED FOR WEEKDAYS, SHOULD BE FIXED
PUN_timeseries = final_result_weekdays.to_numpy()
time_window = len(PUN_timeseries[:, 1])

