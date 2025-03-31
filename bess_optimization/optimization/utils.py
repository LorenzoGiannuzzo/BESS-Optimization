"""

BESS Optimization using NSGA-III Algorithm

    __author__ = "Lorenzo Giannuzzo"
    __maintainer__ = "Lorenzo Giannuzzo"
    __email__ = "lorenzo.giannuzzo@polito.it"
    __status__ = "in progress"
    __version__ = "v0.2.1"
    __license__ = "MIT"

Last Update of current code: 24/03/2025 """
import json
import statistics
import ExcelOpener_l
import Interpolator_l
from pymoo.config import Config

Config.warnings['not_compiled'] = False

class Get_data:
    @staticmethod
    def get_data(file_path2, sheetname3):
        # Import data from an Excel file and a specific sheet using a method from the ExcelOpener module.
        data = ExcelOpener_l.import_file.load_excel(file_path2, sheetname3)
        # Return the imported data.
        return data

class BESS:

    @staticmethod
    def get_bess(technology, properties, se_sp, size):
        # Select specific columns from the properties DataFrame.
        BESS_Parameters = properties.iloc[:, [0, 1, 7, 9, 14, 15]]
        # Filter the DataFrame for the specified technology.
        BESS_Parameters = BESS_Parameters[BESS_Parameters['Technology'] == technology]
        # Further filter the DataFrame for the specified specific energy/specific power.
        BESS_Parameters = BESS_Parameters[BESS_Parameters['Specific Energy / Specific Power'] == se_sp]
        # Add a new column for the size in kWh.
        BESS_Parameters['Size [kWh]'] = size
        # Select specific columns to keep in the final DataFrame.
        BESS_Parameters = BESS_Parameters.iloc[:, [0, 1, 2, 3, 6]]

        # Return the filtered and modified DataFrame.
        return BESS_Parameters

    @staticmethod
    def get_c_d_functions(load_curve):

        # Select charge rate DataFrame from the first 356 rows of load_curve.
        charge_rate = load_curve.iloc[:356, [0, 3.5]]

        # Select discharge rate DataFrame from rows 357 onwards of load_curve.
        discharge_rate = load_curve.iloc[357:, [0, 4, 5]]

        # Interpolate data for the charge rate.
        charge_interpolator = Interpolator_l.DataInterpolator(charge_rate, 'SoC [%]', 'Charge Rate [kWh/(kWhp*h)]')
        charge_rate_interpolated_func = charge_interpolator.interpolate()

        # Interpolate data for the discharge rate.
        discharge_interpolator = Interpolator_l.DataInterpolator(discharge_rate, 'SoC [%]', 'Discharge Rate [kWh/(kWhp*h)]')
        discharge_rate_interpolated_func = discharge_interpolator.interpolate()

        # Return the interpolated functions for charge and discharge rates.
        return charge_rate_interpolated_func, discharge_rate_interpolated_func


def CustomSampling(energy_price, max_charge, max_discharge):

    import numpy as np

    energy_price = np.array(energy_price)
    mean_price = np.mean(energy_price)  # Use numpy's mean
    n = len(energy_price)

    # Initialize X with double length
    X = np.zeros(2 * n)

    # First half: Charge/discharge logic
    for i in range(n):
        if energy_price[i] > mean_price * 1.2:
            X[i] = max_discharge
        elif energy_price[i] < mean_price * 0.8:
            X[i] = max_charge
        else:
            X[i] = 0.0

    # Second half: All 1.0 values
    X[n:] = 1.0

    return X

import pandas as pd
import json
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

def get_season(month):
    """Function to determine the season based on the month."""
    if month in [12, 1, 2]:
        return 'Winter'
    elif month in [3, 4, 5]:
        return 'Spring'
    elif month in [6, 7, 8]:
        return 'Summer'
    else:
        return 'Autumn'


def process_data(consumption_file_path, pv_file_path, pun_file_path):
    # Process consumption data
    df_consumption = pd.read_excel(consumption_file_path, decimal=',', parse_dates=['Data'])
    df_consumption['Month'] = df_consumption['Data'].dt.month
    df_consumption['Season'] = df_consumption['Month'].apply(get_season)
    df_consumption['Hour'] = pd.to_datetime(df_consumption['time'], format='%H:%M:%S').dt.hour
    typical_days_consumption = df_consumption.groupby(['Season', 'Hour'])['value'].mean().unstack()

    output_file_path = r'C:\Users\lorenzo.giannuzzo\PycharmProjects\BESS-Optimization\data\Loads\typycal_days_consumption.xlsx'  # Modify the path as needed
    typical_days_consumption.to_excel(output_file_path)

    df = pd.read_excel(output_file_path)
    df = pd.melt(df, id_vars='Season', value_vars=df.iloc[:, 1:], var_name='Hour')
    df = df.sort_values(by=['Season', 'Hour'], ignore_index=True)
    df.to_excel(output_file_path, index=False)

    # Process PV data
    df_pv = pd.read_csv(pv_file_path, sep=';', parse_dates=['time'], dayfirst=True)
    df_pv.columns = ['time', 'P']
    df_pv['time'] = pd.to_datetime(df_pv['time'], format='%Y%m%d:%H%M')
    df_pv['Month'] = df_pv['time'].dt.month
    df_pv['Season'] = df_pv['Month'].apply(get_season)
    df_pv['Hour'] = df_pv['time'].dt.hour

    # Adjust the PV power values
    df_pv['P'] = (df_pv['P'] / 1000) * 15  # Divide by 1000 and multiply by 15

    typical_days_pv = df_pv.groupby(['Season', 'Hour'])['P'].mean().unstack()

    # Change the output file path to save as CSV
    output_file_path_pv = r'C:\Users\lorenzo.giannuzzo\PycharmProjects\BESS-Optimization\data\Input\pv\typycal_days_pv.csv'  # Modify the path as needed
    typical_days_pv.to_csv(output_file_path_pv)  # Save as CSV instead of Excel

    df = pd.read_csv(output_file_path_pv)  # Read the CSV file
    df = pd.melt(df, id_vars='Season', value_vars=df.iloc[:, 1:], var_name='time')  # Change 'Hour' to 'time'
    df = df.sort_values(by=['Season', 'time'], ignore_index=True)
    df.rename(columns={'value': 'P'}, inplace=True)  # Change 'value' to 'P'
    df.to_csv(output_file_path_pv, index=False)  # Save the melted DataFrame back to CSV

    # PROCESS PUN DATA
    with open(pun_file_path, 'r') as f:  # Open the file in read mode
        pun_data = json.load(f)

    df_pun = pd.DataFrame(pun_data)
    df_pun['datetime'] = pd.to_datetime(df_pun['datetime'].str.replace('Z', ''), utc=True)

    df_pun['Month'] = df_pun['datetime'].dt.month
    df_pun['Season'] = df_pun['Month'].apply(get_season)
    df_pun['Hour'] = df_pun['datetime'].dt.hour

    # Calculate average PUN prices for each hour and season
    typical_days_pun = df_pun.groupby(['Season', 'Hour'])['value'].mean().unstack()

    # Save typical days to JSON
    output_file_path_pun = (r'C:\Users\lorenzo.giannuzzo\PycharmProjects\BESS-Optimization\data\Input\prices\typical_pun.json')

    # Prepare the data for JSON output
    output_data = []
    for season in typical_days_pun.index:
        for hour in typical_days_pun.columns:
            output_data.append({
                "datetime": f"{season} Hour {hour}:00",
                "value": typical_days_pun.loc[season, hour],
                "source": "processed"
            })

    # Save the output data to a JSON file
    with open(output_file_path_pun, 'w') as f:
        json.dump(output_data, f, indent=4)

    # Create subplots
    seasons = typical_days_consumption.index
    num_seasons = len(seasons)

    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(12, 10))
    axes = axes.flatten()  # Flatten the 2D array of axes to 1D for easy iteration

    # Define gradient colormap for consumption (violet to orange)
    consumption_cmap = LinearSegmentedColormap.from_list("violet_orange", ["violet", "orange"])

    # Define gradient colormap for PV production (blue gradient)
    pv_cmap = LinearSegmentedColormap.from_list("blue_gradient", ["lightblue", "darkblue"])

    colors = {
        'Winter': 'lightgrey',
        'Spring': 'lightgrey',
        'Summer': 'lightgrey',
        'Autumn': 'lightgrey'
    }

    for i, season in enumerate(seasons):
        axes[i].fill_between(typical_days_pv.columns,
                             0, typical_days_pv.loc[season],
                             color='orange', alpha=0.4)
        # Fill the area under the consumption curve with a violet to orange gradient (without the line)
        axes[i].fill_between(typical_days_consumption.columns,
                             0, typical_days_consumption.loc[season],
                             color=colors[season], alpha=0.7)
        # Fill the area under the PV curve with a blue gradient (without the line)

        axes[i].set_title(f'{season} Typical Day')
        axes[i].set_xlabel('Hour [h]')
        axes[i].set_ylabel('Power [kW]')
        axes[i].set_xticks(range(0, 24))
        axes[i].grid(True, linestyle='--', alpha=0.3)
        axes[i].legend(['PV production', 'Consumption'], loc='upper right')

        # Rotate x-axis labels by 0 degrees (horizontal labels)
        axes[i].tick_params(axis='x', rotation=0)

    plt.tight_layout()

    # Save the combined plot
    plt.savefig(
        r'C:\Users\lorenzo.giannuzzo\PycharmProjects\BESS-Optimization\Plots\General\combined_typical_days_plot.png',
        dpi=500)

process_data(
    r"C:\Users\lorenzo.giannuzzo\PycharmProjects\BESS-Optimization\data\Loads\BTA6_5.XLSX",
    r'C:\Users\lorenzo.giannuzzo\PycharmProjects\BESS-Optimization\data\Input\pv\year_PV.csv',
    r'C:\Users\lorenzo.giannuzzo\PycharmProjects\BESS-Optimization\data\Input\prices\year_pun.json'
)
