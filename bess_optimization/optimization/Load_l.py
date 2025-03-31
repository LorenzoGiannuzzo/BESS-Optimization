from argparser_l import input_load  # Importing input load parameter
import ExcelOpener_l  # Module for opening Excel files
import pandas as pd  # Data manipulation and analysis

# LOAD EXCEL FILE AND SPECIFIC SHEET
df = ExcelOpener_l.import_file.load_excel(input_load, 'Sheet1')

# CONVERT DATAFRAME TO NUMPY ARRAY AND EXTRACT SPECIFIC COLUMN
data = df.to_numpy()  # Convert DataFrame to NumPy array
data = data[:, 2]  # Extract the 5th column (index 4) from the array

# IN CASE YOU WANT TO SIMULATE ONLY PURE TRADING
#data = np.zeros(len(data))
