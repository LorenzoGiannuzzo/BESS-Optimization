# BESS Optimization For Energy Trading
### BESS Optimization using NSGA-III.

Overview This repository contains the code and data for optimizing Battery Energy Storage Systems (BESS) using evolutionary algorithms. The project focuses on maximizing revenues by optimizing charge and discharge cycles based on PV power plants production and the buying/selling price for electrical energy from/to the grid.

## Installation - Clone the repository: 

``git clone https://github.com/LorenzoGiannuzzo/BESS-Optimization.git cd BESS-Optimization``

Create a virtual environment and activate it:

``python -m venv venv source venv/bin/activate   # On Windows, use `venv\Scripts\activate``

install the required packages:

`pip install -r requirements.txt`

Run the main script to start the optimization process:

`python main.py --input_json <absolute_path_to_input_json> --output_json <absolute_path_to_output_json> --technology <BESS_technology> --size <BESS_size_in_kWh>`
### Input:

- `python main.py`: Executes the main Python script that performs the BESS optimization.
- `--input_json <absolute_path_to_input_json>`: Specifies the absolute path to the input JSON file containing the PUN timeseries data.
- `--output_json <absolute_path_to_output_json>`: Specifies the absolute path where the output JSON file will be saved.
- `--technology <BESS_technology>`: Defines the technology type of the Battery Energy Storage System (e.g., "Li-ion" for Lithium-ion).
- `--size <BESS_size_in_kWh>`: Specifies the size of the Battery Energy Storage System in kilowatt-hours (kWh).
- `--power <BESS_nominal_power_in_kW>`: Specifies the nominal power of the Battery Energy Storage System in kW.
- `--soc <SoC_at_step_0>`: Specifies the initial state of charge of the Battery Energy Storage System in %.
- `--dod <Depth_of_Discharge>`: Specifies the range of SoC in %.
- `--PV_power <Peak_power_in_kW>`: Define the peak power of the PV plants connected to the BESS. Default values is 0 kW. 
- `--POD_power <POD_power_in_kW>`: Define the power of the POD that is responsible for energy exchanges between the system and the electrical grid. Default values is 100 kW. 
- `--n_cycles <number_of_cycles>`: Define the number of cycles the battery already sustained before the optimization. Default value is 0. 

### Output:

The output consists in a .json file structured for each timestep of the considered time_window as follows:\
{\
\
        datetime: --- Expressed as the following format "2024-05-19T22:00:00+00:00Z".\
        PUN: --- Expressed in EUROs/kWh.\
        soc: --- State Of Charge (SoC) expressed as a factor between 0 and 1.\
        c_d_energy: --- Energy which is charged/discharged expressed in kWh. Negative if discharged, positive if charged.\
        Nominal C-rate --- Nominal Charge/Discharge velocity, expressed as % of the total capacity of the BESS.
        revenues: --- Cash flow resulted from charging/discharging energy from t BESS, expressed in EUROs.
        technology: --- Typology of BESS.\
        size: --- Total capacity of the battery, expressed in kWh.\
        dod: --- Depth of Discharge, expressed as "Min_SoC - Max_SoC".\
        n_cycles --- Updated number of cycles battery has after charging or discharging energy.\
        energy_charged_from_pv --- Energy taken from pv and used to charge the BESS, expressed in kWh.\
        energy_taken_from_grid --- Energy taken from the electrical grid and used to charge the BESS, expressed in kWh.\
        energy_sold_from_pv --- Energy produced by PV plants and directly sold to the electrical grid, expressed in kWh.\
        energy_sold_from_BESS --- Energy discharged from the BESS and sold the electrical grid, expressed in kWh.\

}
### How to call main execution - Example:

`python main.py --type Short --input_json C:\Users\lorenzo.giannuzzo\PycharmProjects\BESS-Optimization\Input
\year_pun.json --input_PV C:\Users\lorenzo.giannuzzo\PycharmProjects\BESS-Optimization\Input\year_PV.csv --output_json C:\Users\lorenzo.giannuzzo\PycharmProjects\BESS-Optimization\Output\Short_Simulation\output_weekdays.json --technology Li-ion --size 2500 --power_energy 0.125 --soc 60 --dod 0-100  --PV_power 1000 --POD_power 1300 --n_cycles 1000
`

## Project Structure

The project is structured so that there is a main.py that is called from the command line, and depending on the din punt parameters requested, it executes one of the scripts: main_s.py and main_l.py. 
The main_s.py file runs the annual "short" simulation for 12 typical weekday days and 12 typical weekend days (1 typical day for each month). This simulation is intended to provide a
The main_l.py file runs the "long" simulation, i.e. the optimization of 8760 charge/discharge values.
(yearly simulation with hourly resolution). Both files refer to other specific Python files named "_l" or "_s" depending on the main to which they refer. The files related to the simulations
The files related to the "short" simulations are in the "short simulation" folder, while those related to the "long" simulations are in the "long simulation" folder. Since these are substantially different simulations, they use different
use different input data processing files and different hyperparameter configuration files.

Each of the long or short simulation directory contains the following files:

- `BESS_model.py`: Contains the BESS model definition and related functions.
- `Economic_parameters.py`: Defines economic parameters and functions related to the PUN timeseries.
- `ExcelOpener.py`: Utility for opening and reading Excel files.
- `Interpolator.py`: Contains interpolation functions for data processing.
- `Optimizer.py`: Defines the optimization algorithm and its parameters.
- `Plots.py`: Functions for plotting the results.
- `argparser.py`: Handles command-line arguments and initiates the main optimization process.
- `configuration.py`: Configuration settings for the project.
- `main.py`: The main script that runs the optimization and post-processing.
- `objective_function.py`: Defines the objective function for the optimization.
- `utils.py`: Utility functions used across the project.
- `PV.py`: Import data related to the PV size and production.