from logger import setup_logger
from argparser_l import flexibility_path
import json
import os
import sys

def read_flexibility_json(file_path):
    if not os.path.isabs(file_path):
        print("Error: The provided path is not absolute.", file=sys.stderr)
        sys.exit(1)
    if not os.path.isfile(file_path):
        print(f"Error: File does not exist at the path: {file_path}", file=sys.stderr)
        sys.exit(1)
    try:
        with open(file_path, "r", encoding="utf-8") as file:
            data = json.load(file)
    except json.JSONDecodeError as e:
        print(f"Error: Failed to parse JSON file - {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Error: Unable to read file - {e}", file=sys.stderr)
        sys.exit(1)
    if not isinstance(data, list):
        print("Error: JSON content is not a list/array as expected.", file=sys.stderr)
        sys.exit(1)
    print("Successfully read and parsed JSON content:")
    for item in data:
        # Extracting values
        period = item.get("period", "")
        price = item.get("price", 0.0)
        power = item.get("power", 0)
        # Splitting the period into start and end
        start_period, end_period = period.split(" - ")
        # Formatting the output

    return start_period, end_period, price, power


if flexibility_path != 0.0:
    start_period, end_period, price, power = read_flexibility_json(flexibility_path)
else:
    start_period = 0.0
    end_period = 0.0
    price = 0.0
    power = 0.0
