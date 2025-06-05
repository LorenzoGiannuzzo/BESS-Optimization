# IMPORTING REQUIRED LIBRARIES
from argparser_s import input_load, rec_load  # Importing input load parameter
from Economic_parameters_s import time_window
import ExcelOpener_s  # Module for opening Excel files
import pandas as pd  # Data manipulation and analysis
import numpy as np

if input_load != 0.0:

    # LOAD EXCEL FILE AND SPECIFIC SHEET
    df = ExcelOpener_s.import_file.load_excel(input_load, 'Sheet 1')

    # CONVERT 'Data' COLUMN TO DATETIME FORMAT
    df['Data'] = pd.to_datetime(df['Data'])

    # FILTER DATAFRAME FOR ROWS CORRESPONDING TO THE 15TH DAY OF EACH MONTH
    df_15 = df[df['Data'].dt.day == 15]

    # EXTRACT 'value' COLUMN FROM FILTERED DATAFRAME
    values_15 = df_15['value']

    # CONVERT DATAFRAME TO NUMPY ARRAY AND EXTRACT SPECIFIC COLUMN
    data = df_15.to_numpy()  # Convert DataFrame to NumPy array
    data = data[:, 4]  # Extract the 5th column (index 4) from the array

    # IN CASE YOU WANT TO SIMULATE ONLY PURE TRADING
    #data = np.zeros(len(data))

else:

    data = np.zeros(time_window)


if rec_load != 0.0:
    df_rec = ExcelOpener_s.import_file.load_excel(rec_load, 'Sheet 1')

    # CONVERT 'Data' COLUMN TO DATETIME FORMAT
    df_rec['Data'] = pd.to_datetime(df_rec['Data'])

    # FILTER DATAFRAME FOR ROWS CORRESPONDING TO THE 15TH DAY OF EACH MONTH
    df_15_rec = df_rec[df_rec['Data'].dt.day == 15]

    # EXTRACT 'value' COLUMN FROM FILTERED DATAFRAME
    values_15_rec = df_15_rec['value']

    # CONVERT DATAFRAME TO NUMPY ARRAY AND EXTRACT SPECIFIC COLUMN
    data_rec = df_15_rec.to_numpy()  # Convert DataFrame to NumPy array
    data_rec = data_rec[:, 4]  # Extract the 5th column (index 4) from the array

    data_rec = pd.to_numeric(data_rec)

    data_rec = np.array(data_rec)

else:

    data_rec = np.zeros(time_window)
