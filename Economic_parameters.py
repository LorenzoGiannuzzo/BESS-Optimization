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

# Create a new DataFrame to hold the mean values for each hour of the week
# Group by month and hour, then calculate the mean
df['month'] = df.index.month
df['hour'] = df.index.hour

# Group by month and hour, then calculate the mean
mean_hourly = df.groupby(['month', 'hour'])['value'].mean().reset_index()

# Create a new DataFrame to hold the final result
final_result = pd.DataFrame()

# Iterate over each month to create the final structure
for month in range(1, 13):  # Months from 1 to 12
    month_data = mean_hourly[mean_hourly['month'] == month]
    # Append the month data to the final result
    final_result = pd.concat([final_result, month_data[['hour', 'value']]], ignore_index=True)

# Reset index to get hours as a column
final_result.reset_index(drop=True, inplace=True)

# Convert to numpy array
PUN_timeseries_reduced = final_result.to_numpy()

# Extract optimization time window
time_window = len(PUN_timeseries_reduced[:, 1])

PUN_timeseries = PUN_timeseries_reduced

