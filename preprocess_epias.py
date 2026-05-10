import pandas as pd
from hijridate import Gregorian
from pathlib import Path
from eptr2 import EPTR2
import os

# --- SETUP ---
ROOT_DIR = Path(__file__).resolve().parents[2] 
RAW_DIR = ROOT_DIR / "data" / "raw"
PROCESSED_DIR = ROOT_DIR / "data" / "processed"
PROCESSED_DIR.mkdir(exist_ok=True, parents=True)
dotenv_path = ROOT_DIR / ".env"

eptr = EPTR2(
    use_dotenv=True, 
    dotenv_path=str(dotenv_path), 
    recycle_tgt=True
)

# FETCH 2017 BUFFER DATA
try:
    buffer_df = eptr.call("rt-cons", start_date="2017-12-01", end_date="2017-12-31").rename(columns={'consumption': 'actual_load'})
except Exception as e:
    print(f"Error fetching 2017 data: {e}")
    buffer_df = pd.DataFrame()

# LOAD EXISTING 2018-2025 DATA
df_main  = pd.read_csv(RAW_DIR / "electricity_consumption_2018_2025.csv").rename(columns={'consumption': 'actual_load'})

# CONCATENATE
df = pd.concat([buffer_df, df_main], ignore_index=True)

# TIMEZONE & SORTING

# Aligning timezones
df['timestamp'] = pd.to_datetime(df['date'], utc=True)
df = df.drop(columns=['date', 'time']).sort_values('timestamp').reset_index(drop=True)

# DATA CLEANING
df['actual_load'] = df['actual_load'].interpolate(method='linear')

# HIJRI / RAMADAN LOGIC
def get_hijri_info(ts):
    local_ts = ts.tz_convert('Europe/Istanbul')
    # Use hijridate library
    h = Gregorian(local_ts.year, local_ts.month, local_ts.day).to_hijri()
    
    is_ramadan = 1 if h.month == 9 else 0
    is_eid = 1 if (h.month == 10 and h.day <= 3) else 0
    day_of_ramadan = h.day if h.month == 9 else 0
    return is_ramadan, is_eid, day_of_ramadan

hijri_results = df['timestamp'].apply(get_hijri_info)
df[['is_ramadan', 'is_eid', 'day_of_ramadan']] = pd.DataFrame(hijri_results.tolist(), index=df.index)

# CALENDAR & LAGS & ROLLING
df['hour'] = df['timestamp'].dt.hour
df['day_of_week'] = df['timestamp'].dt.dayofweek
df['month'] = df['timestamp'].dt.month

print("Calculating Lags and Rolling Windows")
df['load_lag_24h'] = df['actual_load'].shift(24)
df['load_lag_168h'] = df['actual_load'].shift(168)

s = df['actual_load'].shift(1)
df['load_rolling_mean_24h'] = s.rolling(window=24).mean()
df['load_rolling_std_24h'] = s.rolling(window=24).std()
df['load_rolling_mean_168h'] = s.rolling(window=168).mean()
df['load_rolling_std_168h'] = s.rolling(window=168).std()

# SLICE BUFFER AND SAVE
df = df[df['timestamp'] >= '2018-01-01 00:00:00+00:00'].copy()

df.to_csv(PROCESSED_DIR / "epias_processed_final.csv", index=False)