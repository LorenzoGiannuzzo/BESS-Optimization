# IMPORTING REQUIRED LIBRARIES
from argparser_l import input_load, rec_load  # Importing input load parameter
from Economic_parameters_l import time_window
import ExcelOpener_l  # Module for opening Excel files
import pandas as pd  # Data manipulation and analysis
import numpy as np

if input_load != 0.0:

    # LOAD EXCEL FILE AND SPECIFIC SHEET
    df = ExcelOpener_l.import_file.load_excel(input_load, 'Sheet 1')

    # CONVERT 'Data' COLUMN TO DATETIME FORMAT
    df['Data'] = pd.to_datetime(df['Data'])

    # CONVERT DATAFRAME TO NUMPY ARRAY AND EXTRACT SPECIFIC COLUMN
    data = df.to_numpy()  # Convert DataFrame to NumPy array
    data = data[:, 4]  # Extract the 5th column (index 4) from the array

    # IN CASE YOU WANT TO SIMULATE ONLY PURE TRADING
    #data = np.zeros(len(data))

else:

    data = np.zeros(time_window)


if rec_load != 0.0:
    df_rec = ExcelOpener_l.import_file.load_excel(rec_load, 'Sheet 1')

    # CONVERT 'Data' COLUMN TO DATETIME FORMAT
    df_rec['Data'] = pd.to_datetime(df_rec['Data'])

    # CONVERT DATAFRAME TO NUMPY ARRAY AND EXTRACT SPECIFIC COLUMN
    data_rec = df_rec.to_numpy()  # Convert DataFrame to NumPy array
    data_rec = data_rec[:, 4]  # Extract the 5th column (index 4) from the array
    data_rec = pd.to_numeric(data_rec)
    data_rec = np.array(data_rec)

else:

    data_rec = np.zeros(time_window)
