# src/create_session_sequences.py (Corrected with Padding)
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import joblib
import pytz
from datetime import datetime, time, timedelta
import argparse

# --- Argument Parser ---
parser = argparse.ArgumentParser(description="Download financial data from MetaTrader 5.")
parser.add_argument('--hypothesis', type=str, default="hyp_a", help="Hypothesis (e.g., 'hyp_a').")
parser.add_argument('--symbol', type=str, default="XAUUSD", help="Financial symbol to download (e.g., 'XAUUSD').")
parser.add_argument('--year', type=str, default="2015", help="Year to start downloading data from.")
parser.add_argument('--timeframe', type=str, default="H1", help="Timeframe for the data (e.g., 'M15', 'H1', 'D1').")
args = parser.parse_args()

# --- Settings ---
hyp = args.hypothesis.lower()
symbol = args.symbol.lower()
timeframe = args.timeframe.lower()
year = args.year

print("Loading and preparing data...")


# --- Load and Pre-process Data ---
df_raw = pd.read_parquet(f'../data/raw/{symbol}_{timeframe}_{year}_present.parquet')
df_raw.set_index('time', inplace=True)
df = df_raw.tz_localize('UTC')

features_to_use = ['open', 'high', 'low', 'close', 'tick_volume']
data = df[features_to_use]


scaler = StandardScaler()
scaled_data_df = pd.DataFrame(scaler.fit_transform(data), index=data.index, columns=data.columns)
joblib.dump(scaler, f'../models/{hyp}_{symbol}_{timeframe}_{year}_pytorch_session_scaler.joblib')
print("Data scaled and scaler saved.")


# --- Create Session-based Sequences with PADDING ---
print("Creating session-based sequences with padding...")
london_tz = pytz.timezone('Europe/London')
X_sequences = []
y_class_targets = []
y_reg_targets = []

# DEFINE OUR FIXED SEQUENCE LENGTH
# From observation, a full session is about 10 hours.
SEQUENCE_LENGTH = 10
NUM_FEATURES = len(features_to_use)

for day in scaled_data_df.index.normalize().unique():
    try:
        # --- Define DYNAMIC London Session ---
        london_open_local = london_tz.localize(datetime.combine(day, time(8, 0)))
        london_close_local = london_tz.localize(datetime.combine(day, time(17, 0)))
        london_open_utc = london_open_local.astimezone(pytz.utc)
        london_close_utc = london_close_local.astimezone(pytz.utc)

        # --- Define Asian Session ---
        previous_day = day - timedelta(days=1)
        asia_part1 = scaled_data_df.loc[str(previous_day.date())].between_time('22:00', '23:59')
        asia_part2 = scaled_data_df.loc[str(day.date())].between_time('00:00', '07:59')
        asia_session_df = pd.concat([asia_part1, asia_part2])

        london_session_df = df[(df.index >= london_open_utc) & (df.index < london_close_utc)]
        
        # --- PADDING LOGIC ---
        # Instead of a rigid check, we handle all non-empty sessions
        if not asia_session_df.empty and not london_session_df.empty:
            
            sequence_data = asia_session_df.values
            current_length = len(sequence_data)
            
            # Create a "canvas" of zeros with our desired final shape
            padded_sequence = np.zeros((SEQUENCE_LENGTH, NUM_FEATURES))
            
            # Copy the actual data into the END of the canvas
            # This is called "pre-padding" and is the standard method.
            padded_sequence[-current_length:] = sequence_data
            
            # Now, `padded_sequence` is guaranteed to be shape (10, 5)
            X_sequences.append(padded_sequence)
            
            # --- Calculate Targets (this logic is unchanged) ---
            london_open = london_session_df['open'].iloc[0]
            london_close = london_session_df['close'].iloc[-1]
            
            london_direction = 1 if london_close > london_open else 0
            y_class_targets.append(london_direction)
            
            london_return = (london_close - london_open) / london_open
            y_reg_targets.append(london_return)

    except Exception as e:
        continue

# Convert lists to numpy arrays
X = np.array(X_sequences)
y_class = np.array(y_class_targets)
y_reg = np.array(y_reg_targets)

print(f"Session sequences created successfully.")
print(f"Shape of X: {X.shape}") # Should be (num_days, 10, 5)
print(f"Shape of y_class: {y_class.shape}")

# --- Save the new sequence data ---
np.save(f'../data/processed/{hyp}_{symbol}_{timeframe}_{year}_session_sequences_X_padded.npy', X)
np.save(f'../data/processed/{hyp}_{symbol}_{timeframe}_{year}_session_targets_y_class.npy', y_class)
np.save(f'../data/processed/{hyp}_{symbol}_{timeframe}_{year}_session_targets_y_reg.npy', y_reg)

print("Padded session sequence data saved.")