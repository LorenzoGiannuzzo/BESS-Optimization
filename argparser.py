import argparse


# Creare il parser degli argomenti
parser = argparse.ArgumentParser(description='Script per l\'ottimizzazione del BESS.')

# Default values
input_json_default = r"C:\Users\lorenzo.giannuzzo\PycharmProjects\BESS-Optimization\Input\pun2.json"
output_json_default =  r"C:\Users\lorenzo.giannuzzo\PycharmProjects\BESS-Optimization\Output\output.json"
technology_default = "Li-ion"
size_default = 2500
power_default = size_default/10
soc_default = 0.2
dod_default = "10-90"


# Aggiungere gli argomenti
parser.add_argument('--input_json', type=str, required=False,default=input_json_default,
                    help='Absolute paht of the file .json for PUN values as input')
parser.add_argument('--output_json', type=str, required=False,default=output_json_default, help='Absolute path of the output file .json')
parser.add_argument('--technology', type=str, required=False, default=technology_default, help='BESS Technology')
parser.add_argument('--size', type=float, required=False,default=size_default, help='BESS Size in kWh')
parser.add_argument('--power', type=float, required=False,default=power_default, help='BES Nominal power in kW')
parser.add_argument('--soc', type=float, required=False,default=soc_default, help='Soc at step 0 of the BESS in %')
parser.add_argument('--dod', type=str, required=False,default=dod_default, help='SoC range in %')
parser.add_argument('--minimize_C', action='store_true', help='Boolean, False = you dont want to minimize c/d velocity, True = you wan to minimize c/d velocity')


# Parsing degli argomenti
args = parser.parse_args()

# Carica i parametri dai comandi
input_json_path = args.input_json
output_json_path = args.output_json
technology = args.technology
size = args.size
soc = args.soc
range_str = args.dod
minimize_C = args.minimize_C

start_str, end_str = range_str.split('-')

soc_min = float(start_str) / 100
soc_max = float(end_str) / 100
