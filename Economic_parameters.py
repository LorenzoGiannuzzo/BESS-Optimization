import pandas as pd
from argparser import input_json_path

#json_file_path = input_json_path

#df = pd.read_json(json_file_path)

# Divide the 'value' column by 1,000,000
#df['value'] = df['value'] / 1000000

# Calculate how many times we need to repeat the data
#num_timesteps = 8760
#current_timesteps = len(df)
#repeat_factor = num_timesteps // current_timesteps
#remaining_timesteps = num_timesteps % current_timesteps

# Repeat the dataframe the necessary number of times
#df_repeated = pd.concat([df] * repeat_factor + [df.iloc[:remaining_timesteps]], ignore_index=True)

# Convert to numpy array if needed
#PUN_timeseries = df_repeated.to_numpy()

json_file_path = input_json_path

df = pd.read_json(json_file_path)

df['value'] = df['value'] / 1000000

PUN_timeseries = df

PUN_timeseries = PUN_timeseries.to_numpy()

time_window = len(PUN_timeseries[:,1])

