from argparser import input_load  # Importing input load parameter
import ExcelOpener  # Module for opening Excel files
import pandas as pd  # Data manipulation and analysis
import os

# LOAD EXCEL FILE AND SPECIFIC SHEET
df = ExcelOpener.import_file.load_excel(input_load, 'Sheet1')

# CONVERT 'Data' COLUMN TO DATETIME FORMAT
df['Data'] = pd.to_datetime(df['Data'])

# CONVERT 'time' COLUMN TO TIMEDATA FORMAT
df['Time'] = pd.to_timedelta(df['time'])  # Assuming 'time' is in HH:MM:SS format

# COMBINE 'Data' AND 'Time' INTO A SINGLE DATETIME COLUMN
df['DateTime'] = df['Data'] + df['Time']

# Check the length of the DataFrame to determine if it's a leap year dataset
if len(df) == 8784:  # Assuming 8784 rows indicates a leap year
    # Filter out February 29th
    df = df[~((df['Data'].dt.month == 2) & (df['Data'].dt.day == 29))]

# Remove duplicate DateTime entries
df = df.drop_duplicates(subset='DateTime', keep='first')  # Keep the first occurrence of each DateTime

# CONVERT DATAFRAME TO NUMPY ARRAY AND EXTRACT SPECIFIC COLUMN
data = df.to_numpy()  # Convert DataFrame to NumPy array
data = data[:, 4]  # Extract the 5th column (index 4) from the array

# IN CASE YOU WANT TO SIMULATE ONLY PURE TRADING
#data = np.zeros(len(data))
