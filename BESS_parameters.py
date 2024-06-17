from utils import Get_data
from utils import BESS

# FILE PATH DATA
file_path = "BESS Data.xlsx"
file_path2 = "PUN.xlsx"
sheetname = "BESS Properties"
sheetname2 = "Li-ion ChargeDischarge Curve 10"
sheetname3 = "PUN"

'''
DEFINE BESS PARAMETERS:
   
   1) Technology
   2) Nominal Capacity / Nominal Power (Specific Energy / Specific Power)
   3) Size

'''

# BESS Parameters

technology = "Li-ion"     #Define BESS Technology
se_sp = 10   #Define Nominal Capacity / Nominal Power - Parameter
size = 2500  #Define BESS size (Nominal Capacity) in kWh

# LOADING FILES

properties = Get_data.get_data(file_path, sheetname)
PUN = Get_data.get_data(file_path2, sheetname3)
load_curve = Get_data.get_data(file_path, sheetname2)


# TAKE CHARGE AND DISCHARGE BATTERY LOAD CURVES (to do: still to be automatized)

charge_rate = load_curve.iloc[:356, [0, 3, 5]]
discharge_rate = load_curve.iloc[357:, [0, 4, 5]]


# GET CHOSEN BESS PARAMETERS

BESS_Parameters = BESS.get_bess(technology, properties, se_sp, size)


# INTERPOLATE CHARGE AND DISCHARGE FUNCTIONS FROM CHARGE AND DISCHARGE RATE

charge_rate_interpolated_func, discharge_rate_interpolated_func = BESS.get_c_d_functions(load_curve)

