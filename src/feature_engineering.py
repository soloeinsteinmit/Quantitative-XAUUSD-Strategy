import pandas as pd
import pandas_ta as ta
import pytz
from datetime import datetime, time, timedelta

# --- Step 1: Load Data and Set Timezone ---
print('Loading raw data...')
year = 2015
# Load the hourly data downloaded earlier using fastparquet(pip install this to allow pandas to use for the data loading)
df_raw = pd.read_parquet(f'../data/raw/xauusd_h1_{year}_present.parquet')

# The 'time' column is our master key. We make it the index of the Dataframe.
df_raw.set_index('time', inplace=True)


"""
Localize the naive index to UTC

This is to deal with forex timezones, standard timezone and Daylight Saving Time(DST) timezones affecting the London & New York time.

CRITICAL: Tell pandas that the existing timestamps are in UTC.
This doesn't change the numbers, it just adds the context.
"""
df = df_raw.tz_localize('UTC')

print(f"Data loaded successfully. Index timezone is now: {df.index.tz}")

# --- Step 2: Pre-calculate Technical Indicators ---
print("\nStep 2: Calculating technical indicators...")

# We use the .ta extension provided by the pandas_ta library.
# The `append=True` argument tells it to add the new columns directly to our DataFrame.
# `length` = lookback period (# of candles).

# Exponential Moving Average, EMA: trend strength clue (are we far from the mean?).
df.ta.ema(length=50, append=True)
df.ta.ema(length=200, append=True)

# Relative Single Index, RSI: momentum exhaustion clue (are we stretched?).
df.ta.rsi(length=14, append=True)

# Average Time Range, ATR: volatility momentum clue (is volatility expanding or contracting?)
df.ta.atr(length=14, append=True)

""" 
- length = lookback period (# of candles).
- EMA50/200 → short vs long trend.
- RSI14 → momentum/overbought/oversold.
- ATR14 → volatility.
- MACD + Bollinger aren’t strictly necessary, but worth experimenting with later for potential edge.

"""

print("Indicators calculated. New columns added.")
df.head() # Use .tail() to see the latest calculated values


# --- Step 3: Loop Through Each Day to Engineer Features ---
print("\nStep 3: Starting daily feature engineering loop...")

# This list will hold the dictionary for each day's calculated data.
daily_data_list = []

# Define the local timezones we need.
london_tz = pytz.timezone('Europe/London')
# ny_tz = pytz.timezone("America/New_York") # for New York


# df.index.date gives us just the date part (e.g., 2025-09-20)
# We group by this to process one day at a time.
for day in df.index.normalize().unique(): # do 10 for now to visualise data
    # .normalize() sets the time to 00:00:00, ensuring one entry per day
    # .unique() gets each of those unique days.
    
    # We'll put all our logic for a single day inside this 'try...except' block.
    # This prevents the whole script from crashing if one day has bad data (e.g., a holiday).
    
    try:
        # --- 3a. Define the DYNAMIC London Session for this specific day ---
        # Create the start time: 8:00 AM in London on the current 'day'.
        london_open_local = london_tz.localize(datetime.combine(day, time(8, 0)))
        # Create the end time: 5:00 PM (17:00) in London on the current 'day'.
        london_close_local = london_tz.localize(datetime.combine(day, time(17, 0)))

        # Now, ask the question: "What time was that in UTC?"
        london_open_utc = london_open_local.astimezone(pytz.utc)
        london_close_utc = london_close_local.astimezone(pytz.utc)
        
        
        # --- 3b. Define the Asian Session (this is the trickiest part) ---
        # The Asian session starts the evening before.
        previous_day = day - timedelta(days=1)
        
        # print(f'day = {day}, timedelata(day=1) = {timedelta(days=1)}')
        # print(f'previous day = {previous_day}')
        
        # Get the evening hours from the previous day (e.g., 22:00 and 23:00) 
        asia_part1 = df.loc[str(previous_day.date())].between_time('22:00', '23:59') 
        # print(f'asia part 1 = {asia_part1}')
        
        # Get the morning hours from the current day (e.g., 00:00 to 07:59) 
        asia_part2 = df.loc[str(day.date())].between_time('00:00', '07:59') 
        # print(f'asia part 2 = {asia_part2}')
        
        # Stitch them together into one session.
        asia_session = pd.concat([asia_part1, asia_part2])
        # print(f'asia session = {asia_session.head(50)}')
        
        
        # --- 3c. Get the data slices (the "Guard Clause") ---
        london_session = df[(df.index >= london_open_utc) & (df.index < london_close_utc)]
        # print(f'london session = {london_session.head(50)}')

        # If either session is empty (e.g., weekend, holiday), skip this day.
        if asia_session.empty or london_session.empty:
            continue
        
        
        # # --- 3d. Calculate FEATURES from the Asian Session ---
        asia_open = asia_session['open'].iloc[0] # First price of the session
        asia_close = asia_session['close'].iloc[-1] # Last price
        asia_high = asia_session['high'].max()
        asia_low = asia_session['low'].min()

        asia_return = (asia_close - asia_open) / asia_open
        asia_range = asia_high - asia_low

        # Get the exact timestamp for the end of the Asian session to look up indicators
        end_of_asia_timestamp = asia_session.index[-1]
        # print("your end here ->", end_of_asia_timestamp)

        # Look up the pre-calculated indicator values at that specific time
        atr_at_asia_close = df.loc[end_of_asia_timestamp]['ATRr_14']
        rsi_at_asia_close = df.loc[end_of_asia_timestamp]['RSI_14']
        ema50_val = df.loc[end_of_asia_timestamp]['EMA_50']
        ema200_val = df.loc[end_of_asia_timestamp]['EMA_200']
        
        ema50_dist = (asia_close - ema50_val) / ema50_val # how are they calculating this though, curious
        ema200_dist = (asia_close - ema200_val) / ema200_val # and this guy
        
        
        # --- 3e. Calculate TARGETS from the London Session ---
        london_open = london_session['open'].iloc[0]
        london_close = london_session['close'].iloc[-1]
        
        # Target for Classification (Hypothesis A1)
        london_direction = 1 if london_close > london_open else 0 # close > open mean a bullish trend and visa versa
        
        # Target for Regression (Hypothesis A2)
        london_return = (london_close - london_open) / london_open # esitmate the return for the overall London session
        
        # --- 3f. Store everything in a dictionary ---
        daily_data_list.append({
            'date': day.date(),
            'day_of_week': day.dayofweek, # Monday=0, Sunday=6
            'asia_return': asia_return,
            'asia_range': asia_range,
            'atr_at_asia_close': atr_at_asia_close,
            'rsi_at_asia_close': rsi_at_asia_close,
            'ema50_dist': ema50_dist,
            'ema200_dist': ema200_dist,
            'london_direction': london_direction, # Our first target
            'london_return': london_return        # Our second target
        })
        
        
    except Exception as e:
        # If anything goes wrong for a specific day, print it and continue.
        print(f"Could not process {day.date()}: {e}")
        continue # Move to the next day
    
print(f"Loop finished. Processed {len(daily_data_list)} trading days.")


# --- Step 4: Assemble Final DataFrame and Save ---
print("\nStep 4: Assembling and saving final feature DataFrame...")

# This is a very efficient way to create a DataFrame from our loop's results.
final_df = pd.DataFrame(daily_data_list)

# Before we set the index, we convert the 'date' column from generic 'object'
# to a proper 'datetime64' format that Parquet understands.
final_df['date'] = pd.to_datetime(final_df['date'])
final_df.set_index('date', inplace=True)

# The first few days of our data won't have EMA_200 values, because it needs 200 hours of data to "warm up".
# .dropna() removes these rows with missing values, ensuring our data is clean for the model.
final_df.dropna(inplace=True)

# year = 2015
# Save the final, clean dataset. We'll use this for training our models.
output_path = f'../data/processed/hyp_a_features_{year}_present.parquet'
final_df.to_parquet(output_path)

print(f"Successfully saved feature data to {output_path}")
print("--- Final DataFrame Info ---")
final_df.info()
print("\n--- Final DataFrame Head ---")
print(final_df.head())