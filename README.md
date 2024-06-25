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

### Example:

`python main.py --input_json C:\path\to\input\pun.json --output_json C:\path\to\output\output.json --technology "Li-ion" --size 2500`

This command runs the optimization script using a Lithium-ion battery with a size of 2500 kWh, taking input data from `pun.json` and saving the results in `output.json`.

The results, including the state of charge (SoC) and revenue calculations, are summarized in a .json file saved in the specified "absolute_path_to_output_json" absolute path, given as input through command line.

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