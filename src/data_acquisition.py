import MetaTrader5 as mt5
import pandas as pd
from datetime import datetime
import sys
import os
from pathlib import Path
import argparse

# --- Argument Parser ---
parser = argparse.ArgumentParser(description="Download financial data from MetaTrader 5.")
parser.add_argument('--symbol', type=str, default="XAUUSD", help="Financial symbol to download (e.g., 'XAUUSD').")
parser.add_argument('--year', type=int, default=datetime.now().year, help="Year to start downloading data from.")
parser.add_argument('--timeframe', type=str, default="H1", help="Timeframe for the data (e.g., 'M15', 'H1', 'D1').")
args = parser.parse_args()

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


def get_timeframe(timeframe : str):
    """
    Takes a string of time frame and returns timeframe from metatrade
    
    Parameters:
        timeframe: string timeframe in uppercase
    
    Returns:
        Metatrade timefram
    """
    TIMEFRAMES : dict = {
        "M1": mt5.TIMEFRAME_M1,
        "M2": mt5.TIMEFRAME_M2,
        "M3": mt5.TIMEFRAME_M3,
        "M4": mt5.TIMEFRAME_M4,
        "M5": mt5.TIMEFRAME_M5,
        "M6": mt5.TIMEFRAME_M6,
        "M10": mt5.TIMEFRAME_M10,
        "M12": mt5.TIMEFRAME_M12,
        "M15": mt5.TIMEFRAME_M15,
        "M20": mt5.TIMEFRAME_M20,
        "M30": mt5.TIMEFRAME_M30,
        "H1": mt5.TIMEFRAME_H1,
        "H2": mt5.TIMEFRAME_H2,
        "H3": mt5.TIMEFRAME_H3,
        "H4": mt5.TIMEFRAME_H4,
        "H6": mt5.TIMEFRAME_H6,
        "H8": mt5.TIMEFRAME_H8,
        "H12": mt5.TIMEFRAME_H12,
        "D1": mt5.TIMEFRAME_D1,
        "W1": mt5.TIMEFRAME_W1,
        "MN1": mt5.TIMEFRAME_MN1,
    }
    timeframe = timeframe.upper() # convert timeframe to upper before passing
    return TIMEFRAMES.get(timeframe, mt5.TIMEFRAME_H1)


# --- Data Download Settings ---
symbol = args.symbol
timeframe = get_timeframe(args.timeframe)  # 1-hour timeframe
year = args.year
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
    file_name = f'{save_path}/{symbol.lower()}_{args.timeframe.lower()}_{year}_present.parquet'
    df.to_parquet(file_name)
    print(f"Data saved to {file_name}")
    print(df.head())