"""

BESS Optimization using NSGA-III Algorithm

    __author__ = "Lorenzo Giannuzzo"
    __maintainer__ = "Lorenzo Giannuzzo"
    __email__ = "lorenzo.giannuzzo@polito.it"
    __status__ = "in progress"
    __version__ = "v0.2.1"
    __license__ = "MIT"

Last Update of current code: 18/11/2024 - 17:06

"""

# IMPORTING LIBRARIES

import argparse
import sys

# CREATE PARSER

parser = argparse.ArgumentParser(description='Script for BESS Optimization.')

# SET DEFAULT VALUES

input_json_default = r"C:\Users\lorenzo.giannuzzo\PycharmProjects\BESS-Optimization\Input\pun2.json"
input_PV_default = r"C:\Users\lorenzo.giannuzzo\PycharmProjects\BESS-Optimization\Input\PV_power.csv"
output_json_default = r"C:\Users\lorenzo.giannuzzo\PycharmProjects\BESS-Optimization\Output\output.json"
technology_default = "Li-ion"
size_default = 2500
power_energy_default = 0
soc_default = 0.2
dod_default = "10-90"
pv_power_default = 0
pod_power_default = 100
n_cycles_default = 0

# GET PARSER ARGUMENTS FROM COMMAND LINE

parser.add_argument('--input_json', type=str, required=False,default=input_json_default,
                    help='Absolute path of the file .json for PUN values as input')

parser.add_argument('--input_PV', type=str, required=False,default=input_PV_default,
                    help='Absolute path of the output file .json')

parser.add_argument('--output_json', type=str, required=False,default=output_json_default,
                    help='Absolute path of the output file .json')

parser.add_argument('--technology', type=str, required=False, default=technology_default,
                    help='BESS Technology')

parser.add_argument('--size', type=float, required=False,default=size_default, help='BESS Size in kWh')

parser.add_argument('--power_energy', type=float, required=False,default=power_energy_default,
                    help='ratio between nominal power and nominal energy capacity')

parser.add_argument('--soc', type=float, required=False,default=soc_default,
                    help='Soc at step 0 of the BESS in %')

parser.add_argument('--dod', type=str, required=False,default=dod_default, help='SoC range in %')

parser.add_argument('--PV_power', type=float, required=False,default=pv_power_default,
                    help='PV peak power')

parser.add_argument('--POD_power', type=float, required=False,default=pod_power_default,
                    help='POD power')

parser.add_argument('--n_cycles', type=float, required=False,default=n_cycles_default,
                    help='number of cycles previously done by teh battery')

parser.add_argument('--weekends', type= str,default = 'True', help='Execute main for weekends')


# ARGUMENTS PARSING

args = parser.parse_args()

# GET PARAMETERS

input_json_path = args.input_json
input_PV = args.input_PV
output_json_path = args.output_json
technology = args.technology
size = args.size
soc = args.soc / 100
range_str = args.dod
PV_power = args.PV_power
power_energy = args.power_energy
BESS_power = size * args.power_energy
POD_power = args.POD_power
n_cycles = args.n_cycles
weekends = args.weekends

start_str, end_str = range_str.split('-')
soc_min = float(start_str) / 100
soc_max = float(end_str) / 100

args2 = [
    sys.executable,  # Percorso all'interprete Python
    'main.py',  # Nome del file principale
    '--type', str('Short'),
    '--input_json', input_json_path,
    '--input_PV', input_PV,
    '--output_json', output_json_path,
    '--technology', technology,
    '--size', str(size),
    '--power_energy', str(power_energy),
    '--soc', str(soc * 100),  # Converti di nuovo in percentuale
    '--dod', range_str,
    '--PV_power', str( str(PV_power)),
    '--POD_power', str(POD_power),
    '--n_cycles', str(n_cycles),
    '--weekends', 'False'
]


