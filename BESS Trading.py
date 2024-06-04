import pandas as pd
import numpy as np
import ExcelOpener


class BESS_Trader:

   #Get BESS Data from Excel Dataframe

   file_path = "BESS Data.xlsx"
   sheetname = "BESS Properties"
   df = ExcelOpener.import_file.load_excel(file_path, sheetname)





