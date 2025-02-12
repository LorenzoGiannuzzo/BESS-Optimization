import pandas as pd
import numpy as np
import os

loads_path_1 = r"data/loads/BTA6_5.xlsx"
file_path_1 = os.path.join("..","..",loads_path_1)

loads_path_2 = r"data/loads/BTA4_7.xlsx"
file_path_2 = os.path.join("..","..",loads_path_2)

loads_path_3 = r"data/loads/BTA5_8.xlsx"
file_path_3 = os.path.join("..","..",loads_path_3)

df_1 = pd.read_excel(file_path_1, 'Sheet 1')
df_2 = pd.read_excel(file_path_2, 'Sheet 1')
df_3 = pd.read_excel(file_path_3, 'Sheet 1')

col_1 = df_1.iloc[:, 4] * 5
col_2 = df_2.iloc[:, 4] * 10
col_3 = df_3.iloc[:, 4] * 7

df = df_1
df.iloc[:,4] = col_1 + col_2 + col_3

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

#print(len(data))
