from utils import Get_data
from BESS_parameters import file_path, file_path2, sheetname, sheetname2, sheetname3


# DEFINE OPTIMIZATION TIME WINDOW
time_window = 72
time_window_e = time_window

# IMPORT PUN TIMESERIES

data = Get_data.get_data(file_path2, sheetname3)
PUN_timeseries = data.iloc[:time_window_e, 2].to_numpy()


