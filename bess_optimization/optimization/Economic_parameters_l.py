""" BESS Optimization using NSGA-III Algorithm

    __author__ = "Lorenzo Giannuzzo"
    __maintainer__ = "Lorenzo Giannuzzo"
    __email__ = "lorenzo.giannuzzo@polito.it"
    __status__ = "in progress"
    __version__ = "v0.2.1"
    __license__ = "MIT"

Last Update of current code: 29/05/2025 """

# IMPORT LIBRARIES
import pandas as pd
from argparser_l import  input_sellprice_path, input_buyprice_path
from Load_l import season
import pandas as pd
from argparser_l import input_sellprice_path, input_buyprice_path
from Load_l import season  # Assuming 'season' is a string representing the season to filter by

# FUNCTION TO LOAD AND FILTER JSON DATA
def load_and_filter_json(file_path, season):
    """Load a JSON file and filter it by the specified season."""
    try:
        df = pd.read_json(file_path)
        df['value'] = df['value'] / 1000000  # Convert value to millions

        # Filter the DataFrame based on the 'datetime' field
        #filtered_df = df[df['datetime'].str.contains(season)]
        return df
    except FileNotFoundError:
        print(f"Error: The file '{file_path}' was not found.")
        return None
    except ValueError as e:
        print(f"Error reading the JSON file '{file_path}': {e}")
        return None
    except Exception as e:
        print(f"An error occurred: {e}")
        return None

# SET PATH FILE FOR SELL PRICE
json_file_path_sell = input_sellprice_path
PUN_timeseries_sell = load_and_filter_json(json_file_path_sell, season)

# Check if the DataFrame is not None before converting to numpy
if PUN_timeseries_sell is not None:
    PUN_timeseries_sell = PUN_timeseries_sell.to_numpy()

    # EXTRACTING OPTIMIZATION TIME WINDOW
    time_window = len(PUN_timeseries_sell[:, 1])
    #print(f"Time window for sell prices: {time_window}")

# SET PATH FILE FOR BUY PRICE
json_file_path_buy = input_buyprice_path
PUN_timeseries_buy = load_and_filter_json(json_file_path_buy, season)

# Check if the DataFrame is not None before converting to numpy
if PUN_timeseries_buy is not None:
    PUN_timeseries_buy = PUN_timeseries_buy.to_numpy()
    #print(f"Loaded buy price timeseries with shape: {PUN_timeseries_buy.shape}")
