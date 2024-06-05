import pandas as pd
import numpy as np
import ExcelOpener
import matplotlib

class BESS_Trader:

   #Get BESS Data from Excel Dataframe

   file_path = "BESS Data.xlsx"
   sheetname = "BESS Properties"
   sheetname2 = "Li-ion ChargeDischarge Curve"
   Properties = ExcelOpener.import_file.load_excel(file_path, sheetname)
   load_curve = ExcelOpener.import_file.load_excel(file_path, sheetname2)









