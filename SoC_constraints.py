import ExcelOpener
import Interpolator

file_path = "BESS Data.xlsx"
file_path2 = "PUN.xlsx"
sheetname = "BESS Properties"
sheetname2 = "Li-ion ChargeDischarge Curve 10"
sheetname3 = "PUN"
# Load properties from file using ExcelOpener
Properties = ExcelOpener.import_file.load_excel(file_path, sheetname)
load_curve = ExcelOpener.import_file.load_excel(file_path, sheetname2)
PUN = ExcelOpener.import_file.load_excel(file_path2, sheetname3)
# Select charge_rate and discharge_rate DataFrames from load_curve
charge_rate = load_curve.iloc[:356, [0, 3.5]]
discharge_rate = load_curve.iloc[357:, [0, 4,5]]

''' BESS SETUP
    - Technology
    - Size [kWh]
    - Specific Energy / Specific Power
    - Initial SoC
    - Self Dischage [%/a]
    - Efficiency
    - Specific Energy / Specific Power
'''

technology = "Li-ion"
se_sp = 10 # Specific Energy / Specific Power
size = 2500 #kWh
SoC_0 = 0.20 #[%] Initial State of Charge
# Extract relevant columns from Properties DataFrame and filter based on technology and specific energy/specific power
BESS_Parameters = Properties.iloc[:,[0,1,7,9,14,15]]
BESS_Parameters = BESS_Parameters[BESS_Parameters['Technology'] == technology]
BESS_Parameters = BESS_Parameters[BESS_Parameters['Specific Energy / Specific Power'] == se_sp]
# Calculate Size [kWh] based on Upper Bound Specific Energy [Wh/kg] and weight, then select relevant columns
BESS_Parameters['Size [kWh]'] = size
BESS_Parameters = BESS_Parameters.iloc[:,[0,1,2,3,6]]

''' OPTIMIZATION PROBLEM STATEMENT
    Optimizaiton Problem Definitizion:
     - Optimization Algorithm (NSGA-III)
     - Dynamic Parameter Evaluation
'''

class SoC_constraints:
   # Interpolate data for charge_rate and discharge_rate
   charge_interpolator = Interpolator.DataInterpolator(charge_rate, 'SoC [%]', 'Charge Rate [kWh/(kWhp*h)]')
   charge_rate_interpolated_func = charge_interpolator.interpolate()
   discharge_interpolator = Interpolator.DataInterpolator(discharge_rate, 'SoC [%]', 'Discharge Rate [kWh/(kWhp*h)]')
   discharge_rate_interpolated_func = discharge_interpolator.interpolate()

