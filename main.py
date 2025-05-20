import asyncio
import csv
import os

import requests
from kasa import Discover
from pushbullet import Pushbullet
from datetime import datetime, timedelta
import logging
from logging.handlers import RotatingFileHandler
from dotenv import load_dotenv
load_dotenv()

# === CONFIGURATION ===
PLUG_IP = os.environ.get("PLUG_IP")
PUSHBULLET_API_KEY = os.environ.get("PUSHBULLET_API_KEY")
POWER_THRESHOLD = 0.005  # Watts
POWER_DURATION_ALERT_SEC = 60  # seconds
NO_POWER_DURATION_ALERT_SEC = 6 * 60 * 60  # 6 hours
RAIN_ALERT_THRESHOLD_MM = 10.0  # 10 mm in past 24 hours
LATITUDE = float(os.environ.get("LATITUDE"))   # Replace with your location
LONGITUDE = float(os.environ.get("LONGITUDE"))

LOG_DIR = "logs"
ALERT_LOG_FILE = os.path.join(LOG_DIR, "power_monitor.log")
DATA_LOG_FILE = os.path.join(LOG_DIR, "power_data.csv")
RAIN_LOG_FILE = os.path.join(LOG_DIR, "rain_log.csv")

# === STATE TRACKERS ===
power_on_start = None
last_power_time = datetime.now()
alerted_no_power = False
alerted_long_power = False
last_rain_check_time = datetime.min
last_weekly_rain_check_time = datetime.min
rain_last_24h = 0.0

pb = Pushbullet(PUSHBULLET_API_KEY)

# === Logging Setup ===
os.makedirs(LOG_DIR, exist_ok=True)

log_handler = RotatingFileHandler(
    ALERT_LOG_FILE, maxBytes=1024 * 1024, backupCount=5  # 1MB max, keep 5 backups
)
log_formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
log_handler.setFormatter(log_formatter)

logger = logging.getLogger()
logger.setLevel(logging.INFO)
logger.addHandler(log_handler)

def send_alert(message):
    log_message = f"Alert: {message}"
    logger.warning(log_message)
    pb.push_note("Power Monitor Alert", message)

def log_power_data(timestamp, power):
    with open(DATA_LOG_FILE, mode='a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([timestamp.isoformat(), power])

def get_rainfall_last_24h():
    try:
        url = (
            f"https://api.open-meteo.com/v1/forecast?"
            f"latitude={LATITUDE}&longitude={LONGITUDE}"
            "&hourly=rain,soil_moisture_0_to_1cm,soil_moisture_1_to_3cm,soil_moisture_3_to_9cm,soil_moisture_9_to_27cm,soil_moisture_27_to_81cm"
            "&past_hours=24"
            "&forecast_hours=1"
            "&timezone=auto"
        )
#        print(url)
        response = requests.get(url)
        data = response.json()
        hourly = data["hourly"]
        rain_vals = hourly["rain"][-24:]  # last 24 hours
        soil_0 = hourly["soil_moisture_0_to_1cm"][-1]
        soil_1 = hourly["soil_moisture_1_to_3cm"][-1]
        soil_3 = hourly["soil_moisture_3_to_9cm"][-1]
        soil_9 = hourly["soil_moisture_9_to_27cm"][-1]
        soil_27 = hourly["soil_moisture_27_to_81cm"][-1]
        total_rain = sum(rain_vals)
        logger.info(f"Rainfall last 24h: {total_rain:.2f} mm")
        logger.info(f"Soil moisture at 27-81cm: {soil_27:.2f} mm3/mm3")
        return total_rain, soil_0, soil_1, soil_3, soil_9, soil_27
    except Exception as e:
        logger.error(f"Error fetching rainfall data: {e}")
        return 0.0  # assume no rain if it fails

def get_rainfall_last_7_days():
    try:
        url = (
            f"https://api.open-meteo.com/v1/forecast?"
            f"latitude={LATITUDE}&longitude={LONGITUDE}"
            "&daily=precipitation_sum"
            "&past_days=7"
            "&forecast_days=1"
            "&timezone=auto"
        )
        response = requests.get(url)
        data = response.json()
        total_7day_rain = sum(data["daily"]["precipitation_sum"])
        return total_7day_rain
    except Exception as e:
        logger.error(f"Error fetching 7-day rainfall: {e}")
        return 0.0
        
def log_rain_data(timestamp, rainfall_mm, soil_0, soil_1, soil_3, soil_9, soil_27):
    file_exists = os.path.isfile(RAIN_LOG_FILE)
    with open(RAIN_LOG_FILE, mode="a", newline="") as file:
        writer = csv.writer(file)
        if not file_exists:
            writer.writerow(["timestamp", "rainfall_mm", "soil_moisture_0_to_1cm", "soil_moisture_1_to_3cm", "soil_moisture_3_to_9cm", "soil_moisture_9_to_27cm", "soil_moisture_27_to_81cm"])
        writer.writerow([timestamp.isoformat(), f"{rainfall_mm:.2f}", f"{soil_0:.3f}", f"{soil_1:.3f}", f"{soil_3:.3f}", f"{soil_9:.3f}", f"{soil_27:.3f}"])

# Add CSV headers if file doesn't exist
if not os.path.isfile(DATA_LOG_FILE):
    with open(DATA_LOG_FILE, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["timestamp", "power_watts"])

async def monitor_plug():
    global power_on_start, last_power_time, alerted_no_power, alerted_long_power, rain_last_24h, last_rain_check_time, last_weekly_rain_check_time

    dev = await Discover.discover_single(PLUG_IP)
    await dev.update()
    logger.info(f"Monitoring plug at {PLUG_IP}")

    while True:
        try:
            await dev.update()
            power = dev.emeter_realtime['power_mw'] / 1000.0  # convert to Watts
            now = datetime.now()

            # === Check Rainfall once per hour ===
            if (now - last_rain_check_time).total_seconds() > 3600:
                rain_last_24h, soil_0, soil_1, soil_3, soil_9, soil_27 = get_rainfall_last_24h()
                last_rain_check_time = now
                log_rain_data(now, rain_last_24h, soil_0, soil_1, soil_3, soil_9, soil_27)

            # === Check Weekly Rainfall once per Day ===
            # Track last weekly rain check
            # last_weekly_rain_check_time = datetime.min

            # Inside monitor_plug()
            if (now - last_weekly_rain_check_time).total_seconds() > 86400:  # once per day
                total_rain_7d = get_rainfall_last_7_days()
                if total_rain_7d < 38.1:  # 1.5 inches in mm
                    send_alert(f"Lawn watering alert: only {total_rain_7d:.1f} mm rain in past 7 days.")
                last_weekly_rain_check_time = now


            if power > POWER_THRESHOLD:
                log_power_data(now, power)
                if not power_on_start:
                    send_alert(f"Power draw detected: {power:.2f}W")
                    power_on_start = now
                elif (now - power_on_start).total_seconds() > POWER_DURATION_ALERT_SEC and not alerted_long_power:
                    send_alert("Power draw over 1 minute — turning plug OFF.")
                    await dev.turn_off()
                    alerted_long_power = True

                last_power_time = now
                alerted_no_power = False
            else:
                power_on_start = None
                alerted_long_power = False

                # Check for long no-power period AND rain
                time_since_last_power = (now - last_power_time).total_seconds()
                if time_since_last_power > NO_POWER_DURATION_ALERT_SEC and not alerted_no_power:
                    if rain_last_24h >= RAIN_ALERT_THRESHOLD_MM:
                        send_alert(f"No power draw in 6h AND {rain_last_24h:.1f} mm rain in 24h.")
                        alerted_no_power = True
        except Exception as e:
            logger.info(f"[{datetime.now()}] Error communicating with plug: {e}")
        await asyncio.sleep(1)

if __name__ == "__main__":
    asyncio.run(monitor_plug())
