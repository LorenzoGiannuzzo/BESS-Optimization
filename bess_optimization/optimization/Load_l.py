from argparser_l import input_load  # Importing input load parameter
import ExcelOpener_l  # Module for opening Excel files
import pandas as pd  # Data manipulation and analysis

# LOAD EXCEL FILE AND SPECIFIC SHEET

season = "Autumn"

df = ExcelOpener_l.import_file.load_excel(input_load, 'Sheet1')

# CONVERT DATAFRAME TO NUMPY ARRAY AND EXTRACT SPECIFIC COLUMN

df = df[df['Season'] == season]

data = df.to_numpy()

data = data[:, 2]  # Extract the 5th column (index 4) from the array

# IN CASE YOU WANT TO SIMULATE ONLY PURE TRADING
#data = np.zeros(len(data))
