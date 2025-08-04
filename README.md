# BESS Optimization For Energy Trading, Self-Consumption and PV Management
### Optimization Task Performed using NSGA-III.
This repository contains the code and data for optimizing Battery Energy Storage
Systems (BESS) using evolutionary (NSGA-III) algorithms.The project focuses on maximizing revenues
by optimizing charge and discharge cycles based on PV power plants production, the electric load
self-consumption, and the buying/selling price for electrical energy from/to the grid.

# Project Conceptualization
The project is mainly divided in two main subsets:
        1) Long_Simulation
        2) Short_Simulation

1) Long_Simulation: takes as input files = PUN/PZ prices (.json), PV production (.csv files) and 
Load profiles (.xlsx files). Files length (which is the temporal extension of the optimization 
task) must be equal among all 3 files, and can be variable. To test the algorithm, those files are
already available in their specific folders (see project structure),  otherwise, different files
(but with their same structure) can be used by modifying the command line used to call main.py

2) Short_Simulation: takes as input files = PUN/PZ prices (.json), PV production (.csv files)
and Load profiles (.xlsx files). Files length (which is the temporal extension of the optimization 
task) must be 1 year and equal among all 3 files, and can be variable. This subsection of the project
is used to give an estimated value of the system revenues for 1 year, avoiding to execute a run for 
8760 values (1 each hour of the year), but working instead with 1 typical day for each month. To test the algorithm, those files are already available in their specific folders (see project structure),  otherwise, different      
files (but with their same structure) can be used by modifying the command line used to call main.py

# Project Input:
python main.py `--type`: Decide if performing annual simulation with typical days (Long) or simulation with a fixed time window (Short) 
               `--input_pun`: absolute path to PUN prices data in .json format 
               `--input_PV`: absolute path to PV production data in .csv format 
               `--input_load`: absolute path to load data in .xlsx format 
               `--rec_load`: absolute path to rec load data in .xlsx format 
               `--rec_production`: absolute path to REC PV data in .csv format 
               `--flexibility_order`: absolute path to flexibility data in .json format 
               `--output_json`: absolute path to output folder  
               `--technology`: BESS technology 
               `--size`: BESS size in kWh  
               `--power_energy`: Power to Energy ratio 
               `--soc`: Initial State of Charge (%) 
               `--dod`: <Depth_of_Discharge_%_range> 
               `--PV_power`: PV peak power in (kW) 
               `--POD_power`: POD power in (kW) 
               `--n_cycles`: Battery cycles already sustained (float number) 
               `--self_consumption`: True or False, if True it prioritize user's self consumption over everything else.
# Project Output:
The output consists in a .json file structured for each timestep (1h) of the considered time_window as follows. This .json files is save in a specified path and name according to the given input (see Porject Input)

{\
\
        `datetime`:                             # Timestamp of the simulation step  
        `PUN`:                                  # PUN value in €/kWh  
        `soc`:                                  # State of charge  
        `c_d_energy`:                           # Total charged/discharged energy from BESS  
        `Nominal C-rate`:                       # Nominal charge rate  
        `C-rate`:                               # Actual C-rate  
        `revenues`:                             # Total revenues for the time step  
        `rev_BESS`:                             # Revenues from BESS  
        `rev_PV`:                               # Revenues from PV  
        `rev_SC`:                               # Revenues from self-consumption  
        `technology`:                           # BESS technology used  
        `size`:                                 # Size of the BESS  
        `dod`:                                  # Depth of discharge  
        `n_cycles`:                             # Number of cycles  
        `energy_charged_from_PV_to_BESS`:       # Energy charged from PV to BESS  
        `energy_taken_from_grid_to_BESS`:       # Energy taken from the grid to BESS  
        `energy_sold_from_PV`:                  # Energy sold from PV to the grid  
        `energy_sold_from_BESS`:                # Energy sold from BESS to the grid  
        `energy_from_BESS_to_load`:             # Energy from BESS to load  
        `energy_from_PV_to_load`:               # Energy from PV to load  
        `energy_self_consumed`:                 # Energy self-consumed (load)  
        `flexibility_success`:                  # Flag that defines if in that hour the flexibility request was achieved or not (-1 means requested but not achieved, 0 means not requested, 1 means requested and achieved)

}

# Project Installation - Clone the repository: 
``git clone https://github.com/LorenzoGiannuzzo/BESS-Optimization.git cd BESS-Optimization``
Create a virtual environment and activate it:
``python -m venv venv source venv/bin/activate   # On Windows, use `venv\Scripts\activate``
install the required packages:
`pip install -r requirements.txt`

# Project Structure
The project is structured as follows:
.
|── README.md
|── LICENSE
|── requirements.txt
|── main.py                     # Script responsible to handle both long and short-run simulations
|── docs                        # folder containing useful docs or additional documentations
|── Plots                       # folder containing plots of the optimization results
|── data/                       # folder containing input and output data (input and ouput data are specified by filepath when calling main.py from command line)
|   |
|   |── Input/                  # input data 
|   |   |
|   |   |── BESS Data.xlsx      # Excel files containing experimental data required to get battery model
|   |   |── XXXX.csv            # a .csv file containing normalized PV production (1kW) for a specific time window (for example 26h - see file 26h_PV.csv)
|   |   |── XXXX.json           # a .json containing PUN or PZ price for a specific time window (for example 26h - see file 26h_pun.csv)
|   |
|   |── Loads/                  # folder containing electrical consumption files
|   |   |
|   |   |── BTA4_7.xlsx         # synthetic data with nominal power supply 6 kW < P < 10 kW
|   |   |── BTA5_8.xlsx         # synthetic data with nominal power supply 10 kW < P < 16 kW
|   |   |── BTA6_5.xlsx         # synthetic data with nominal power supply P > 16 kW
|   |
|   |── Output/                 # folder optimization output files
|   |   |
|   |   |── Long Simulation/    # specific folder for long-run optimization results
|   |   |   |
|   |   |   |── XXXX.json       # a .json file containing all the outcomes of the optimizaiton task for the long-run simulations
|   |   |
|   |   |── Short Simulation/   # specific folder for short-run optimization results
|   |   |   |
|   |   |   |── XXXX.json       # a .json file containing all the outcomes of the optimizaiton task for the short-run simulations
|   |   
|── bess_optimization/          # base directory for the source code of the project
|   |
|   |── Long Simulation/
|   |   |
|   |   |── BESS_model_l.py     # Battery Model used for the long-run simulation
|   |   |── configuration_l.py  # Configuration parameters definition of the NSGA - III for the long-run simulation
|   |   |── Economic_parameters_l.py       # Script responsible for gathering the economic parameters to perform the optimization task for the long-run simulation
|   |   |── ExcelOpener_l.py    # Script containing functions and classes to open excel files as input parameters for the long-run simulation
|   |   |── Interpolator_l.py      # Script responsible to interpolate functions for the BESS model
|   |   |── main_l.py           # main python script for the execution of the long-run simulation
|   |   |── objective_function_l.py        # Objective function definition for the optimization problem for the long-run simulation
|   |   |── Optimizer_l.py      # Script containing the instance of the optimization problem for the long-run simulation
|   |   |── Plots_l.py          # Script containing the plot classes and functions
|   |   |── PV_l.py             # Script responsible of gathering PV production data for the long-run simulation
|   |   |── Optimizer_l.py      # Script containing the instance of the optimization problem for the long-run simulation
|   |   |── test.py             # Test Scripts 
|   |   |── utils.py            # Scripts used for subprocesses
|   |
|   |── Short Simulation/
|   |   |
|   |   |── argparser_s.py      # Parser that gets the input parameters for the short-run simulation
|   |   |── BESS_model_s.py     # Battery Model used for the short-run simulation
|   |   |── configuration_s.py  # Configuration parameters definition of the NSGA - III for the short-run simulation
|   |   |── Economic_parameters_s.py       # Script responsible for gathering the economic parameters to perform the optimization task for the long-run simulation
|   |   |── ExcelOpener_s.py    # Script containing functions and classes to open excel files as input parameters for the short-run simulation
|   |   |── Interpolator_s.py   # Script responsible to interpolate functions for the BESS model
|   |   |── main_s.py           # main python script for the execution of the short-run simulation
|   |   |── objective_function_s.py        # Objective function definition for the optimization problem for the short-run simulation
|   |   |── Optimizer_s.py      # Script containing the instance of the optimization problem for the short-run simulation
|   |   |── Plots_s.py          # Script containing the plot classes and functions
|   |   |── PV_s.py             # Script responsible of gathering PV production data for the short-run simulation
|   |   |── Optimizer_s.py      # Script containing the instance of the optimization problem for the short-run simulation
|   |   |── test.py             # Test Scripts 
|   |   |── utils.py            # Scripts used for subprocesses
