""" BESS Optimization using NSGA-III Algorithm

    __author__ = "Lorenzo Giannuzzo"
    __maintainer__ = "Lorenzo Giannuzzo"
    __email__ = "lorenzo.giannuzzo@polito.it"
    __status__ = "in progress"
    __version__ = "v0.2.1"
    __license__ = "MIT"

Last Update of current code: 21/05/2025 """

# IMPORTING LIBRARIES --------------------------------------------------------------------------------------------------

import argparse
import logging
import os.path
import sys
from logger import setup_logger

# CREATE PARSER --------------------------------------------------------------------------------------------------------

# LOGGER SETUP
setup_logger()

parser = argparse.ArgumentParser(description='Script for BESS Optimization.')

# SET DEFAULT VALUES
input_json_default = 0.0
input_PV_default = 0.0
input_load_default = 0.0
output_json_default = r"C:\Users\lorenzo.giannuzzo\PycharmProjects\BESS-Optimization\data\Output\output.json"
technology_default = "Li-ion"
size_default = 0.0
power_energy_default = 0
soc_default = 0.2
dod_default = "10-90"
pv_power_default = 0
pod_power_default = 100
n_cycles_default = 0

# GET PARSER ARGUMENTS FROM COMMAND LINE
parser.add_argument('--input_pun', type=str, required=False, default=input_json_default,
                    help='Absolute path of the file .json for PUN values as input')

parser.add_argument('--input_PV', type=str, required=False, default=input_PV_default,
                    help='Absolute path of the output file .json')

parser.add_argument('--input_load', type=str, required=False, default=input_load_default,
                    help='Absolute path of the output file .xlsx')

parser.add_argument('--rec_load', type=str, required=False, default=input_load_default,
                    help='Absolute path of the output file .xlsx')

parser.add_argument('--rec_production', type=str, required=False, default=input_load_default,
                    help='Absolute path of the output file .xlsx')

parser.add_argument('--output_json', type=str, required=False, default=output_json_default,
                    help='Absolute path of the output file .json')

parser.add_argument('--technology', type=str, required=False, default=technology_default,
                    help='BESS Technology')

parser.add_argument('--size', type=float, required=False, default=size_default, help='BESS Size in kWh')

parser.add_argument('--power_energy', type=float, required=False, default=power_energy_default,
                    help='ratio between nominal power and nominal energy capacity')

parser.add_argument('--soc', type=float, required=False ,default=soc_default,
                    help='Soc at step 0 of the BESS in %')

parser.add_argument('--dod', type=str, required=False, default=dod_default, help='SoC range in %')

parser.add_argument('--PV_power', type=float, required=False, default=pv_power_default,
                    help='PV peak power')

parser.add_argument('--POD_power', type=float, required=False, default=pod_power_default,
                    help='POD power')

parser.add_argument('--n_cycles', type=float, required=False, default=n_cycles_default,
                    help='number of cycles previously done by teh battery')

parser.add_argument('--weekends', type= str, required=False, default='False', help='Execute main for weekends')

parser.add_argument('--self_consumption', type=str, required=True, default='False',
                    help='Force the Algorithm to self-consume energy for the load')

# ARGUMENTS PARSING ----------------------------------------------------------------------------------------------------

args = parser.parse_args()

# GET PARAMETERS FROM ARGPARSER

# ZONAL ELECTRICITY PRICES
input_json_path = args.input_pun
assert os.path.exists(input_json_path), logging.error("Electrical Price path file does not exists.\n\n")

# PV PRODUCTION TIME SERIES
input_PV_path = args.input_PV
# assert os.path.exists(input_PV_path), logging.error("PV production path file does not exists.\n\n")

# LOAD TIME SERIES
rec_load = args.rec_load
assert os.path.exists(input_json_path), logging.error("PV production path file does not exists.\n\n")

# REC PRODUCTION
rec_production = args.rec_production
assert os.path.exists(input_json_path), logging.error("PV production path file does not exists.\n\n")

# LOAD TIME SERIES
input_load = args.input_load
assert os.path.exists(input_json_path), logging.error("PV production path file does not exists.\n\n")

# OUTPUT PATH DEFINITION
output_json_path = args.output_json
assert os.path.exists(output_json_path), logging.error("The specified output path file does not exists.\n\n")

# DEFINE BESS TECHNOLOGY
technology = args.technology

# DEFINE BESS SIZE
size = args.size
assert size >= 0, logging.error("Input BESS Size is negative.\n\n")

# INITIAL SOC
soc = args.soc / 100
assert soc >= 0, logging.error("Input Initial SoC is negative.\n\n")

# DoD RANGE
range_str = args.dod

# PV POWER
PV_power = args.PV_power
assert PV_power >= 0, logging.error("Input PV Power is negative.\n\n")

# RATIO NOMINAL POWER / NOMINAL CAPACITY
power_energy = args.power_energy
assert power_energy >= 0, logging.error("Input Ratio Nominal Power / Nominal Capacity is negative.\n\n")

# BESS POWER
BESS_power = size * args.power_energy
assert BESS_power >= 0, logging.error("Input Ratio Nominal Power / Nominal Capacity is negative.\n\n")

# POD POWER
POD_power = args.POD_power
assert BESS_power >= 0, logging.error("Input BESS power is negative.\n\n")

# NUMBER OF CYCLES
n_cycles = args.n_cycles
assert n_cycles >= 0, logging.error("Input Number of cycles is negative.\n\n")

weekends = args.weekends
self_consumption = args.self_consumption

# GET MINIMUM AND MAXIMUM ALLOWED SoC
start_str, end_str = range_str.split('-')

soc_min = float(start_str) / 100
assert soc_min >= 0, logging.error("Minimum Allowed SoC is negative.\n\n")

soc_max = float(end_str) / 100
assert soc_max >= 0, logging.error("Maximum Allowed SoC is negative.\n\n")

if soc_min > soc_max:

    logging.error("Maximum Allowed SoC is lower than Minimum Allowed SoC.\n\n")

    sys.exit()

elif soc_min == soc_max:

    logging.error("Maximum Allowed SoC is equal to Minimum Allowed SoC.\n\n")

    sys.exit()

# Execute Long Simulation with new parameters --------------------------------------------------------------------------
args2 = [
    sys.executable,
    'main.py',
    '--type', str('Short'),
    '--input_json', input_json_path,
    '--input_PV', input_PV_path,
    '--output_json', output_json_path,
    '--technology', technology,
    '--size', str(size),
    '--power_energy', str(power_energy),
    '--soc', str(soc * 100),  # Convert Again into percentage
    '--dod', range_str,
    '--PV_power', str( str(PV_power)),
    '--POD_power', str(POD_power),
    '--n_cycles', str(n_cycles),
    '--weekends', 'False'
]
