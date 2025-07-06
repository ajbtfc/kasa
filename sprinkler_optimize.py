import numpy as np
import requests
import pandas as pd
from scipy.optimize import minimize
import os
from dotenv import load_dotenv
from pushbullet import Pushbullet
import logging
from logging.handlers import RotatingFileHandler
import aiohttp
from pyrainbird import async_client
from pyrainbird.async_client import CreateController, AsyncRainbirdController
import asyncio

load_dotenv()

LOG_DIR = "logs"
LATITUDE = float(os.environ.get("LATITUDE"))   # Replace with your location
LONGITUDE = float(os.environ.get("LONGITUDE"))
DAILY_MAX_WATER = 1.5
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
    "&daily=precipitation_sum"
    "&past_days=7"
    "&forecast_days=14"
    "&timezone=auto"
    "&precipitation_unit=inch"
)
response = requests.get(url)
data = response.json()
df = pd.DataFrame({"date":data['daily']['time'], "rain":data['daily']['precipitation_sum']})
df['sprinkler'] = 0.0
sprinkler_df = pd.read_csv("sprinkler_runs.csv")
sprinkler_df['date'] = pd.to_datetime(sprinkler_df['date'])
df['date'] = pd.to_datetime(df['date'])
df = df.merge(sprinkler_df, on='date', how='left')
df['sprinkler'] = df['amount'].combine_first(df['sprinkler'])
df = df.drop(columns='amount')

# Constants
WINDOW = 7
TARGET = 1.5
N = len(df)
NUM_TO_OPTIMIZE = 14
opt_start_idx = N - NUM_TO_OPTIMIZE

# Objective function: sum of squared errors over 7-day windows
# def objective(x):
    # Copy the sprinkler column
#    s = df['sprinkler'].values.copy()
    # Insert optimization values into the last 7 days
#    s[opt_start_idx:] = x

#    total_error = 0
#    for i in range(N - WINDOW + 1):
#        rain_sum = df['rain'].values[i:i+WINDOW].sum()
#        sprinkler_sum = s[i:i+WINDOW].sum()
#        total = rain_sum + sprinkler_sum
#        error = (total - TARGET) ** 2
#        total_error += error
#    return total_error

# Initial guess: 0.1 inches per day
# x0 = np.full(NUM_TO_OPTIMIZE, 0.1)

# Bounds: sprinkler values between 0 and 1
# bounds = [(0, DAILY_MAX_WATER)] * NUM_TO_OPTIMIZE

# Run the optimization
# result = minimize(objective, x0, bounds=bounds, method='L-BFGS-B')

# Update DataFrame with optimized values
#if result.success:
#    df.loc[opt_start_idx:, 'sprinkler'] = result.x
#else:
#    print("Optimization failed:", result.message)

# Objective function with conditional watering
def objective(x):
    # Apply conditional logic: only water if rain is below 1 inch
    s = np.zeros(N)
    rain = df['rain'].values

    for i in range(opt_start_idx, N):
        # Only add sprinkler water if rain is less than 1 inch
        if rain[i] < DAILY_MAX_WATER:
            s[i] = min(x[i - opt_start_idx], DAILY_MAX_WATER - rain[i])  # Never overfill beyond 1 inch

    total_error = 0
    for i in range(opt_start_idx, N - WINDOW + 1):
        rain_sum = rain[i:i+WINDOW].sum()
        sprinkler_sum = s[i:i+WINDOW].sum()
        total = rain_sum + sprinkler_sum
        error = (total - TARGET) ** 2
        total_error += error
    return total_error

# Initial guess: 0.1 inches per day
x0 = np.full(NUM_TO_OPTIMIZE, 0)

# Bounds: between 0 and max possible fill (1 inch if no rain)
bounds = [(0, DAILY_MAX_WATER)] * NUM_TO_OPTIMIZE

# Run the optimization
result = minimize(objective, x0, bounds=bounds, method='L-BFGS-B')

# Update DataFrame with optimized values
if result.success:
    # Reapply same logic to get final sprinkler values
    sprinkler = np.concatenate([
    df['sprinkler'].values[:opt_start_idx],
    np.zeros(N - opt_start_idx)
])
    rain = df['rain'].values
    for i in range(opt_start_idx, N):
        if rain[i] < DAILY_MAX_WATER:
            sprinkler[i] = min(result.x[i - opt_start_idx], DAILY_MAX_WATER - rain[i])
    df['sprinkler'] = sprinkler
else:
    print("Optimization failed:", result.message)

df['7_day_total'] = df['rain'].rolling(window=7).sum() + df['sprinkler'].rolling(window=7).sum()
# Resulting DataFrame
print(df.head(8))

# Ensure 'time' column is datetime
df['date'] = pd.to_datetime(df['date'])

# Get tomorrow's date
tomorrow = pd.Timestamp.today().normalize() + pd.Timedelta(days=0)

# Filter rows where the date part of 'date' matches tomorrow
df_tomorrow = df[df['date'].dt.normalize() == tomorrow]
print(df_tomorrow)
if df_tomorrow['sprinkler'].iloc[0] + df_tomorrow['rain'].iloc[0] > 1:
    zone_1 = df_tomorrow['sprinkler'].iloc[0] * 60
    zone_2 = zone_1 / 3
    zone_3 = zone_1
    zone_4 = zone_1

    print(f"Sprinkler plan for {tomorrow.date()} - {df_tomorrow['sprinkler'].iloc[0]}")
    print(f"Front yard - {df_tomorrow['sprinkler'].iloc[0]*60} minutes")
    print(f"Back yard - {df_tomorrow['sprinkler'].iloc[0]*120} minutes")
    send_alert(f"""
    Sprinkler plan for {tomorrow.date()} - {df_tomorrow['sprinkler'].iloc[0]}
    Front yard - {df_tomorrow['sprinkler'].iloc[0]*60} minutes
    Back yard - {df_tomorrow['sprinkler'].iloc[0]*120} minutes
    """
           )

    # Example wrapper assuming 'controller' is your AsyncRainbirdController
    async def run_irrigation_sequence(controller):
        zone_minutes = {
            1: zone_1,
            2: zone_2,
            3: zone_3,
            4: zone_4,
        }

        try:
            for zone, minutes in zone_minutes.items():
                print(f"Starting irrigation on zone {zone} for {minutes} minutes...")
                await controller.irrigate_zone(zone, minutes)
                await asyncio.sleep(minutes * 60)  # Wait while it irrigates
                await controller.stop_irrigation()
                print(f"Stopped irrigation on zone {zone}\n")
        except Exception as e:
            print(f"⚠️ Error during irrigation: {e}")
            try:
                await controller.stop_irrigation()
                print("✅ Irrigation stopped due to error.")
            except Exception as stop_error:
                print(f"❌ Failed to stop irrigation: {stop_error}")
            raise  # Optional: re-raise if you want to propagate the error


    new_row = pd.DataFrame([{
        'date': tomorrow,
        'amount': df_tomorrow['sprinkler'].iloc[0]
    }])

    sprinkler_df = pd.concat([sprinkler_df, new_row], ignore_index=True)
    sprinkler_df.to_csv("sprinkler_runs.csv", index=False)
    # Define async irrigation logic
    async def main():
        try:
            async with aiohttp.ClientSession() as client:
                controller = async_client.CreateController(
                    client,
                    "192.168.1.228",
                    "AaBbCc123"
                )
                await run_irrigation_sequence(controller)
        except Exception as e:
            print(f"🚨 Top-level error: {e}")

    # Run it
    asyncio.run(main())
else:
    print(f"Water need only { df_tomorrow['sprinkler'].iloc[0] + df_tomorrow['rain'].iloc[0]} inches")
    send_alert(
        f"No sprinkler today. Sprinkler - {df_tomorrow['sprinkler'].iloc[0]} Rain - {df_tomorrow['rain'].iloc[0]} inches")


#async with aiohttp.ClientSession() as client:
#        controller: AsyncRainbirdController = async_client.CreateController(
#        client,
#        "192.168.1.228",
#        "AaBbCc123"
#    )
#        await run_irrigation_sequence(controller)
#    asyncio.run()
# Read csv
# Add row
# Write to csv
# Return df with past week and next 2 weeks
