"""

BESS Optimization using NSGA-III Algorithm

    __author__ = "Lorenzo Giannuzzo"
    __maintainer__ = "Lorenzo Giannuzzo"
    __email__ = "lorenzo.giannuzzo@polito.it"
    __status__ = "in progress"
    __version__ = "v0.2.1"
    __license__ = "MIT"

Last Update of current code: 06/11/2024 - 17:38

"""

# IMPORT LIBRARIES

import pandas as pd

# CREATE IMPORT_FILE CLASS

# Define CSVHandler class
class import_file:

    def load_excel(file_path, sheetname):

        # Attempt to load the Excel file into a Pandas DataFrame
        try:
            data = pd.read_excel(file_path, sheet_name = sheetname)


        except FileNotFoundError:

            # Handle FileNotFoundError, print an error message

            print(f"Error: The file '{file_path}' was not found.")

        except Exception as e:

            # Handle other exceptions, print an error message with details

            print(f"Error loading the file '{file_path}': {e}")

        return data
