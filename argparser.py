import argparse

# Creare il parser degli argomenti
parser = argparse.ArgumentParser(description='Script per l\'ottimizzazione del BESS.')

# Aggiungere gli argomenti
parser.add_argument('--input_json', type=str, required=True,
                    help='Path assoluto del file .json da cui estrarre la timeseries del PUN')
parser.add_argument('--output_json', type=str, required=True, help='Path assoluto del file .json di output')
parser.add_argument('--technology', type=str, required=True, help='Technology del BESS')
parser.add_argument('--size', type=int, required=True, help='Size del BESS in kWh')

# Parsing degli argomenti
args = parser.parse_args()

# Carica i parametri dai comandi
input_json_path = args.input_json
output_json_path = args.output_json
technology = args.technology
size = args.size