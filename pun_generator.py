import json
import pandas as pd
from datetime import datetime, timedelta

# Load the existing PUN timeseries from pun2.json
with open('Input/50h_pun.json', 'r') as file:
    pun_data = json.load(file)

# Create a list to hold the new timeseries data
new_pun_data = []

# Get the starting datetime from the first entry in the existing data
start_datetime = datetime.fromisoformat(pun_data[0]['datetime'].replace("Z", "+00:00"))

# Generate data for 1 year (365 days)
for i in range(31 * 24):  # 24 hours for each day
    # Calculate the current datetime
    current_datetime = start_datetime + timedelta(hours=i)

    # Get the value from the existing data, cycling through if necessary
    value = pun_data[i % len(pun_data)]['value']

    # Create a new entry
    new_entry = {
        "datetime": current_datetime.isoformat() + "Z",  # Convert to ISO format with Z
        "value": value,
        "source": pun_data[0]['source']  # Assuming the source is the same
    }

    # Append the new entry to the list
    new_pun_data.append(new_entry)

# Save the new timeseries to a new JSON file
with open('Input/month_pun.json', 'w') as outfile:
    json.dump(new_pun_data, outfile, indent=4)