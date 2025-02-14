# IMPORTING REQUIRED LIBRARIES
from argparser_s import input_load  # Importing input load parameter
import ExcelOpener_s  # Module for opening Excel files
import pandas as pd  # Data manipulation and analysis

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

# test