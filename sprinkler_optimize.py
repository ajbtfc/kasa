import numpy as np
import requests
import pandas as pd
from scipy.optimize import minimize
import os
from dotenv import load_dotenv
from pushbullet import Pushbullet
import logging
from logging.handlers import RotatingFileHandler

load_dotenv()

LOG_DIR = "logs"
LATITUDE = float(os.environ.get("LATITUDE"))   # Replace with your location
LONGITUDE = float(os.environ.get("LONGITUDE"))
DAILY_MAX_WATER = .75
PUSHBULLET_API_KEY = os.environ.get("PUSHBULLET_API_KEY")
SPRINKLER_LOG_FILE = os.path.join(LOG_DIR, "sprinkler_log.csv")


pb = Pushbullet(PUSHBULLET_API_KEY)


# === Logging Setup ===
os.makedirs(LOG_DIR, exist_ok=True)

log_handler = RotatingFileHandler(
    SPRINKLER_LOG_FILE, maxBytes=1024 * 1024, backupCount=5  # 1MB max, keep 5 backups
)
log_formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
log_handler.setFormatter(log_formatter)

logger = logging.getLogger()
logger.setLevel(logging.INFO)
logger.addHandler(log_handler)

def send_alert(message):
    log_message = f"Alert: {message}"
    logger.warning(log_message)
    pb.push_note("Lawn Watering Plan", message)

url = (
    f"https://api.open-meteo.com/v1/forecast?"
    f"latitude={LATITUDE}&longitude={LONGITUDE}"
    "&daily=rain_sum"
    "&past_days=7"
    "&forecast_days=14"
    "&timezone=auto"
    "&precipitation_unit=inch"
)
response = requests.get(url)
data = response.json()
df = pd.DataFrame({"date":data['daily']['time'], "rain":data['daily']['rain_sum']})
df['sprinkler'] = 0.0
sprinkler_df = pd.read_csv("kasa/sprinkler_runs.csv")
sprinkler_df['date'] = pd.to_datetime(sprinkler_df['date'])
df['date'] = pd.to_datetime(df['date'])
df = df.merge(sprinkler_df, on='date', how='left')
df['sprinkler'] = df['amount'].combine_first(df['sprinkler'])
df = df.drop(columns='amount')

# Constants
WINDOW = 7
TARGET = 1.5
N = len(df)
NUM_TO_OPTIMIZE = 13
opt_start_idx = N - NUM_TO_OPTIMIZE

# Objective function: sum of squared errors over 7-day windows
def objective(x):
    # Copy the sprinkler column
    s = df['sprinkler'].values.copy()
    # Insert optimization values into the last 7 days
    s[opt_start_idx:] = x

    total_error = 0
    for i in range(N - WINDOW + 1):
        rain_sum = df['rain'].values[i:i+WINDOW].sum()
        sprinkler_sum = s[i:i+WINDOW].sum()
        total = rain_sum + sprinkler_sum
        error = (total - TARGET) ** 2
        total_error += error
    return total_error

# Initial guess: 0.1 inches per day
x0 = np.full(NUM_TO_OPTIMIZE, 0.1)

# Bounds: sprinkler values between 0 and 1
bounds = [(0, DAILY_MAX_WATER)] * NUM_TO_OPTIMIZE

# Run the optimization
result = minimize(objective, x0, bounds=bounds, method='L-BFGS-B')

# Update DataFrame with optimized values
if result.success:
    df.loc[opt_start_idx:, 'sprinkler'] = result.x
else:
    print("Optimization failed:", result.message)

df['7_day_total'] = df['rain'].rolling(window=7).sum() + df['sprinkler'].rolling(window=7).sum()
# Resulting DataFrame
print(df.tail(10))

# Ensure 'time' column is datetime
df['date'] = pd.to_datetime(df['date'])

# Get tomorrow's date
tomorrow = pd.Timestamp.today().normalize() + pd.Timedelta(days=1)

# Filter rows where the date part of 'date' matches tomorrow
df_tomorrow = df[df['date'].dt.normalize() == tomorrow]

print(f"Sprinkler plan for {tomorrow.date()} - {df_tomorrow['sprinkler'].iloc[0]}")
print(f"Front yard - {df_tomorrow['sprinkler'].iloc[0]*60} minutes")
print(f"Back yard - {df_tomorrow['sprinkler'].iloc[0]*120} minutes")
send_alert(f"""
Sprinkler plan for {tomorrow.date()} - {df_tomorrow['sprinkler'].iloc[0]}
Front yard - {df_tomorrow['sprinkler'].iloc[0]*60} minutes
Back yard - {df_tomorrow['sprinkler'].iloc[0]*120} minutes
"""
           )
# Read csv
# Add row
# Write to csv
# Return df with past week and next 2 weeks
