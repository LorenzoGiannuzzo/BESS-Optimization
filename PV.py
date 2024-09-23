import pandas as pd

# File path
file_path = 'input/PV_power.csv'

pv_production = pd.read_csv(file_path, sep=';', usecols=['time', 'P'])

from argparser import PV_power

pv_production['P'] = pv_production['P'] * PV_power / 1000

