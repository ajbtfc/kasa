import pandas as pd
from main import DATA_LOG_FILE, RAIN_LOG_FILE

# Load data
power_df = pd.read_csv(DATA_LOG_FILE, parse_dates=["timestamp"])
power_df['timestamp'] = pd.to_datetime(power_df['timestamp'])
rain_soil_df = pd.read_csv(RAIN_LOG_FILE, parse_dates=["timestamp"])
rain_soil_df['timestamp'] = pd.to_datetime(rain_soil_df['timestamp'])
rain_soil_df = rain_soil_df.sort_values("timestamp").reset_index(drop=True)

# Sort power data
power_df = power_df.sort_values("timestamp").reset_index(drop=True)

# Step 1: Identify distinct pump runs
# If there's more than 30 seconds between rows, we treat it as a new run
power_df['time_diff'] = power_df['timestamp'].diff().dt.total_seconds().fillna(9999)
power_df['new_run'] = power_df['time_diff'] > 30

# Assign a unique run_id to each group
power_df['run_id'] = power_df['new_run'].cumsum()

# Get the *first* timestamp of each run
run_starts = power_df.groupby('run_id')['timestamp'].min().reset_index()
run_starts = run_starts.sort_values('timestamp').reset_index(drop=True)

# Step 2: Build regression rows
rows = []

for i in range(len(run_starts) - 1):
    start_time = run_starts.loc[i, 'timestamp']
    end_time = run_starts.loc[i + 1, 'timestamp']
    duration = (end_time - start_time).total_seconds() / 60  # in minutes

    interval_data = rain_soil_df[(rain_soil_df['timestamp'] > start_time) & (rain_soil_df['timestamp'] <= end_time)]

    if interval_data.empty:
        continue

    row = {
        "start_time": start_time,
        "end_time": end_time,
        "duration_to_next_run_min": duration,
    }

    for col in interval_data.columns:
        if col.startswith("soil_moisture"):
            row[f"avg_{col}"] = interval_data[col].mean()

    rows.append(row)

# Step 3: Save result
regression_df = pd.DataFrame(rows)
regression_df.to_csv("sump_regression_dataset.csv", index=False)
