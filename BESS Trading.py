import pandas as pd
import numpy as np
import ExcelOpener
import pybamm
import matplotlib

class BESS_Trader:

   #Get BESS Data from Excel Dataframe

   file_path = "BESS Data.xlsx"
   sheetname = "BESS Properties"
   Properties = ExcelOpener.import_file.load_excel(file_path, sheetname)



