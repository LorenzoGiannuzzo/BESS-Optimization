import pandas as pd
from argparser_l import input_PV
from Load_l import season

# File path
file_path = input_PV

pv_production = pd.read_csv(file_path, sep=';')

#pv_production = pd.read_csv(file_path, sep=';')

from argparser_l import PV_power

#pv_production = pv_production[pv_production['Season'] == season].reset_index()

pv_production['P'] = pv_production['P'] * PV_power / 1000


