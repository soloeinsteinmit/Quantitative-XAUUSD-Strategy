import MetaTrader5 as mt5
import pandas as pd
from datetime import datetime
import sys
import os
from pathlib import Path

current_dir = Path(os.getcwd())
sys.path.append(str(current_dir.parent))

from src.config import DEMO_ACCOUNT_NUMBER, PASSWORD, SERVER

# --- Connection Settings ---
# Replace with your demo account details
account_number = int(DEMO_ACCOUNT_NUMBER) # accept and int
password = PASSWORD
server = SERVER

# --- Connect to MetaTrader 5 ---
if not mt5.initialize(login=account_number, password=password, server=server):
    print("initialize() failed, error code =", mt5.last_error())
    quit()
print("Connected to MetaTrader 5")


# --- Data Download Settings ---
symbol = "XAUUSD"
timeframe = mt5.TIMEFRAME_H1  # 1-hour timeframe
year = 2020
start_date = datetime(year, 1, 1)
# start_date = datetime(2018, 1, 1)
end_date = datetime.now()

# --- Fetch the data ---
rates = mt5.copy_rates_range(symbol, timeframe, start_date, end_date)

# --- Shutdown connection ---
mt5.shutdown()


# --- Create a Pandas DataFrame ---
save_path = "../data/raw"
if rates is None:
    print("No data received from MT5.")
else:
    df = pd.DataFrame(rates)
    # Convert timestamp to a readable datetime format
    df['time'] = pd.to_datetime(df['time'], unit='s')
    print(f"Data downloaded successfully: {df.shape[0]} rows")
    
    # --- Save the data to a file ---
    # Using Parquet is more efficient than CSV
    df.to_parquet(f'{save_path}/xauusd_h1_{year}_present.parquet')
    print(f"Data saved to xauusd_h1_{year}_present.parquet")
    print(df.head())