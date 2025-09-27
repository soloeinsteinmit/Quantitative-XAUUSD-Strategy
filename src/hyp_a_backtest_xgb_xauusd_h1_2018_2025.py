import pandas as pd
import joblib
from backtesting import Backtest, Strategy
import pytz # We'll need this for our dynamic session times
from datetime import datetime, time

# --- Step 2: Define the Trading Strategy ---
print("Defining the ML-based trading strategy...")

class MLStrategy(Strategy):
    # --- The init() method is called once at the start ---
    def init(self):
        print("Initializing strategy...")
        # 1. Load the pre-trained champion model
        model_path = '../models/xgb_classifier_hyp_a_xauusd_h1_2018_present.joblib'
        self.model = joblib.load(model_path)
        print("Model loaded successfully.")

        # 2. Load the feature data that corresponds to the model
        features_path = '../data/processed/hyp_a_features_xauusd_h1_2018_present.parquet'
        self.features = pd.read_parquet(features_path)
        print("Feature data loaded successfully.")
        
        # 3. Store the London timezone for dynamic open times
        self.london_tz = pytz.timezone('Europe/London')
        self._last_trade_date = None

    # --- The next() method is called for each new candle ---
    def next(self):
        # self.data.index[-1] gives us the timestamp of the current candle
        current_time_utc = self.data.index[-1]
        current_date = current_time_utc.date()
        current_hour_utc = current_time_utc.hour
        
        if self._last_trade_date == current_date:
            return
        
        # --- Determine the London open time for THIS specific day ---
        london_open_local = self.london_tz.localize(datetime.combine(current_date, time(8, 0)))
        london_open_utc = london_open_local.astimezone(pytz.utc)
        target_trade_hour_utc = london_open_utc.hour

        # --- TRADING LOGIC ---
        # We only want to make one decision per day, exactly at the London open.
        if current_hour_utc == target_trade_hour_utc:
            
            # Convert the current_date object to a Pandas Timestamp to match the index type.
            current_date_ts = pd.to_datetime(current_date)
            
            # Defensive check: Make sure we have features for today
            # Now, check if this Timestamp is in our features index.
            if current_date_ts not in self.features.index:
                print(f"Skipping trade on {current_date}: No features found.")
                return
            
            self._last_trade_date = current_date

            # 1. Get today's features
            # We select the row for the current date and drop the target columns
            # Use the Timestamp to locate the features
            today_features = self.features.loc[[current_date_ts]].drop(columns=['london_direction', 'london_return'])
            
            # 2. Use the model to make a prediction
            prediction = self.model.predict(today_features)[0] # [0] to get the single value
            print(f"TRADE SIGNAL on {current_date}: Prediction is {'BULLISH' if prediction == 1 else 'BEARISH'}")
            
            # 3. Execute the trade based on the prediction
            # We will also close any existing position before opening a new one.
            # This ensures we only hold one position at a time, for one day.
            if prediction == 1: # Model predicts Bullish
                self.position.close() # Close any short position from a previous day
                self.buy() # Open a new long position
                
            elif prediction == 0: # Model predicts Bearish
                self.position.close() # Close any long position
                self.sell() # Open a new short position
                
# --- Step 3: Load Data and Run the Backtest (Corrected Version) ---
print("\nPreparing data for backtest...")

# 1. Load the raw hourly price data.
price_data = pd.read_parquet('../data/raw/xauusd_h1_2018_present.parquet')
price_data.set_index('time', inplace=True)
price_data = price_data.tz_localize('UTC')

# --- THE FIX IS HERE ---
# The backtesting.py library requires specific column names with capital letters.
# Let's rename our columns to match its requirements.
price_data.rename(columns={
    'open': 'Open',
    'high': 'High',
    'low': 'Low',
    'close': 'Close',
    'tick_volume': 'Volume' # We'll rename 'tick_volume' to 'Volume'
}, inplace=True)
print("Price data columns renamed to match backtesting.py requirements.")
# --- END OF FIX ---

# 2. Isolate the test period.
train_size_raw = int(len(price_data) * 0.8)
backtest_data = price_data.iloc[train_size_raw:]
print(f"Backtesting on data from {backtest_data.index[0]} to {backtest_data.index[-1]}")

# 3. Configure and initialize the backtest engine
bt = Backtest(
    backtest_data,
    MLStrategy,
    cash=10000,
    commission=.0002,
    exclusive_orders=True
)

# 4. Run the backtest!
print("\nRunning backtest...")
stats = bt.run()
print("Backtest complete.")

# 5. Print the results and generate the plot
print("\n--- Backtest Results ---")
print(stats)

print("\nGenerating equity curve plot...")
bt.plot()