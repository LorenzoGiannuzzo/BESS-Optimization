# BESS Optimization For Energy Trading
### BESS Optimization using NSGA-III.

Overview This repository contains the code and data for optimizing Battery Energy Storage Systems (BESS) using evolutionary algorithms. The project focuses on maximizing revenues by optimizing charge and discharge cycles.

## Installation - Clone the repository: 

``git clone https://github.com/LorenzoGiannuzzo/BESS-Optimization.git cd BESS-Optimization``

Create a virtual environment and activate it:

``python -m venv venv source venv/bin/activate   # On Windows, use `venv\Scripts\activate``

install the required packages:

`pip install -r requirements.txt`

Run the main script to start the optimization process:

`python main.py --input_json <absolute_path_to_input_json> --output_json <absolute_path_to_output_json> --technology <BESS_technology> --size <BESS_size_in_kWh>`
### Explanation:

- `python main.py`: Executes the main Python script that performs the BESS optimization.
- `--input_json <absolute_path_to_input_json>`: Specifies the absolute path to the input JSON file containing the PUN timeseries data.
- `--output_json <absolute_path_to_output_json>`: Specifies the absolute path where the output JSON file will be saved.
- `--technology <BESS_technology>`: Defines the technology type of the Battery Energy Storage System (e.g., "Li-ion" for Lithium-ion).
- `--size <BESS_size_in_kWh>`: Specifies the size of the Battery Energy Storage System in kilowatt-hours (kWh).
- `--power <BESS_nominal_power_in_kW>`: Specifies the nominal power of the Battery Energy Storage System in kW.
- `--soc <SoC_at_step_0>`: Specifies the initial state of charge of the Battery Energy Storage System in %.
- `--dod <Depth_of_Discharge>`: Specifies the range of SoC in %.
- `--minimize_C <Boolean>`: Parameter that requires no values. Default values is FALSE, is it's in command line, it's set to be True. If TRUE, it changes the optimization problem maximizing revenues and minimizing C-rate

### Example:

`python main.py --input_json C:\Users\lorenzo.giannuzzo\PycharmProjects\BESS-Optimization\Input\pun2.json --output_json C:\Users\lorenzo.giannuzzo\PycharmProjects\BESS-Optimization\Output\output.json --technology Li-ion --size 2500 --power 250  --soc 0.9 --dod 20-80
`

## Project Structure

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
- `requirements.txt`: Lists the dependencies required to run the project.
- `utils.py`: Utility functions used across the project.