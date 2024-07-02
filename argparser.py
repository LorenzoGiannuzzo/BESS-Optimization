import argparse

import argparser

# Creare il parser degli argomenti
parser = argparse.ArgumentParser(description='Script per l\'ottimizzazione del BESS.')

# Default values
input_json_default = r"C:\Users\lorenzo.giannuzzo\PycharmProjects\BESS-Optimization\Input\pun2.json"
output_json_default =  r"C:\Users\lorenzo.giannuzzo\PycharmProjects\BESS-Optimization\Output\output.json"
technology_default = "Li-ion"
size_default = 2500
soc_default = 0.2

# Aggiungere gli argomenti
parser.add_argument('--input_json', type=str, required=False,default=input_json_default,
                    help='Path assoluto del file .json da cui estrarre la timeseries del PUN')
parser.add_argument('--output_json', type=str, required=False,default=output_json_default, help='Path assoluto del file .json di output')
parser.add_argument('--technology', type=str, required=False, default=technology_default, help='Technology del BESS')
parser.add_argument('--size', type=float, required=False,default=size_default, help='Size del BESS in kWh')
parser.add_argument('--soc', type=float, required=False,default=soc_default, help='Soc iniziale del BESS in %')

# Parsing degli argomenti
args = parser.parse_args()

# Carica i parametri dai comandi
input_json_path = args.input_json
output_json_path = args.output_json
technology = args.technology
size = args.size
soc = args.soc
