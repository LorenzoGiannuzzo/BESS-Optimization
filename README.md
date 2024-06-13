# BESS Optimization For Energy Trading
### BESS Optimization using NSGA-III.

Overview This repository contains the code and data for optimizing Battery Energy Storage Systems (BESS) using evolutionary algorithms. The project focuses on maximizing revenues by optimizing charge and discharge cycles.
###### Table of Contents 
- [Installation](#installation) 
- [Project Structure](#project-structure) 

## Installation - Clone the repository: 

``git clone https://github.com/LorenzoGiannuzzo/BESS-Optimization.git cd BESS-Optimization``

Create a virtual environment and activate it:

``python -m venv venv source venv/bin/activate   # On Windows, use `venv\Scripts\activate``

install the required packages:

`pip install -r requirements.txt`

Run the main script to start the optimization process:

`python main.py`

The results, including the state of charge (SoC) and revenue calculations, will be printed and plotted.

## Project Structure

- `Optimizer.py`: Contains the `Optimizer` class which defines the optimization algorithm and its parameters.
- `objective_function.py`: Defines the revenue calculation and other related functions.
- `Plots.py`: Contains functions for plotting the results.
- `BESS_parameters.py`: Defines parameters related to the battery energy storage system.
- `Economic_parameters.py`: Defines economic parameters such as the PUN timeseries.
- `Tests/`: Contains test scripts for the project.
- `BESS Data.xlsx` and `PUN.xlsx`: Data files necessary for the optimization process.
- `requirements.txt`: Lists the dependencies required to run the project.