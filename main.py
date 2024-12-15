import argparse
import subprocess
import sys
import os

def run_main_s(args):
    # Prepare the arguments for main_s.py, excluding the --flag argument
    filtered_args = [arg for arg in sys.argv[1:] if arg != '--type' and arg != 'Short']
    short_directory = r"Short Simulation"
    subprocess.run(['python', os.path.join(short_directory, 'main_s.py')] + filtered_args)

def run_main_l(args):
    # Run main_p.py with the provided arguments
    filtered_args = [arg for arg in sys.argv[1:] if arg != '--type' and arg != 'Long']
    short_directory = r"Long Simulation"
    subprocess.run(['python', os.path.join(short_directory, 'main_l.py')] + filtered_args)

if __name__ == "__main__":

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

    parser = argparse.ArgumentParser(description="Choose script to execute")

    # Adding a flag for Yes/No
    parser.add_argument('--type', choices=['Short', 'Long'], required=True,
                        help="Specify 'Yes' to run main_s or 'No' to run main_p")

    parser.add_argument('--input_json', type=str, required=False, default=input_json_default,
                        help='Absolute path of the file .json for PUN values as input')

    parser.add_argument('--input_PV', type=str, required=False, default=input_PV_default,
                        help='Absolute path of the output file .json')

    parser.add_argument('--output_json', type=str, required=False, default=output_json_default,
                        help='Absolute path of the output file .json')

    parser.add_argument('--technology', type=str, required=False, default=technology_default,
                        help='BESS Technology')

    parser.add_argument('--size', type=float, required=False, default=size_default, help='BESS Size in kWh')

    parser.add_argument('--power_energy', type=float, required=False, default=power_energy_default,
                        help='ratio between nominal power and nominal energy capacity')

    parser.add_argument('--soc', type=float, required=False, default=soc_default,
                        help='Soc at step 0 of the BESS in %')

    parser.add_argument('--dod', type=str, required=False, default=dod_default, help='SoC range in %')

    parser.add_argument('--PV_power', type=float, required=False, default=pv_power_default,
                        help='PV peak power')

    parser.add_argument('--POD_power', type=float, required=False, default=pod_power_default,
                        help='POD power')

    parser.add_argument('--n_cycles', type=float, required=False, default=n_cycles_default,
                        help='number of cycles previously done by the battery')

    # ARGUMENTS PARSING
    args = parser.parse_args()

    # Execute the corresponding script based on the flag
    if args.type == 'Short':
        run_main_s(args)
    elif args.type == 'Long':
        run_main_l(args)