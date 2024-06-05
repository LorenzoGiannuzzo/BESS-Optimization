import pandas as pd
import openpyxl

# Define CSVHandler class
class import_file:

    def load_excel(file_path, sheetname):
        # Attempt to load the excel file into a Pandas DataFrame
        try:
            data = pd.read_excel(file_path, sheet_name = sheetname)
            print(f"File '{file_path}' loaded successfully.")
        except FileNotFoundError:
            # Handle FileNotFoundError, print an error message
            print(f"Error: The file '{file_path}' was not found.")
        except Exception as e:
            # Handle other exceptions, print an error message with details
            print(f"Error loading the file '{file_path}': {e}")

        return data
