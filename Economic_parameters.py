from utils import Get_data
from configuration import time_window

file_path2 = "PUN.xlsx"
sheetname3 = "PUN"

# IMPORT PUN TIMESERIES

data = Get_data.get_data(file_path2, sheetname3)
PUN_timeseries = data.iloc[:time_window, 2].to_numpy()


