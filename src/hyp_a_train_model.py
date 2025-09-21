# src/train_model.py
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split # We will explain why NOT to use this
from sklearn.metrics import classification_report, mean_absolute_error, r2_score
import joblib # For saving our trained models
import argparse
from datetime import datetime
import os  # For file operations and checking existing reports
import sys  # For system exit codes

# --- Argument Parser ---
parser = argparse.ArgumentParser(description="Train models on financial data.")
parser.add_argument('--year', type=int, default=2015, help="Year of the data to process.")
parser.add_argument('--hypothesis', type=str, default="hyp_a", help="Hypothesis to use for feature and model naming.")
args = parser.parse_args()

# --- Step 1: Load Feature Data ---
print("Step 1: Loading features...")

try:
    year = args.year
    hypothesis = args.hypothesis
    # First try with the full name pattern
    symbol = 'xauusd'  # default
    timeframe = 'h1'   # default
    feature_path = f'../data/processed/{hypothesis}_features_{symbol}_{timeframe}_{year}_present.parquet'
    
    # If not found, list available feature files to help debugging
    if not os.path.exists(feature_path):
        print(f"Error: Feature file not found: {feature_path}")
        print("\nAvailable feature files in processed directory:")
        try:
            files = [f for f in os.listdir('../data/processed') if f.endswith('.parquet')]
            if files:
                for f in files:
                    print(f"- {f}")
            else:
                print("No .parquet files found!")
        except Exception as e:
            print(f"Could not list directory: {e}")
            
        print(f"\nPlease run: python {hypothesis}_feature_engineering.py --year {year} --symbol {symbol} --timeframe {timeframe}")
        sys.exit(1)  # Exit with error code 1
        
    # Read metadata from feature file name
    feature_file = pd.read_parquet(feature_path)
    
    # Extract timeframe and symbol from the data
    timeframe = feature_file['timeframe'].iloc[0] if 'timeframe' in feature_file.columns else 'h1'
    symbol = feature_file['symbol'].iloc[0] if 'symbol' in feature_file.columns else 'xauusd'
    df = feature_file.drop(columns=['timeframe', 'symbol'], errors='ignore')
except Exception as e:
    print(f"Error loading feature file: {str(e)}")
    print(f"Please ensure the feature engineering step completed successfully.")
    sys.exit(1)  # Exit with error code 1

# Verify required columns exist
required_columns = ['london_direction', 'london_return']
missing_columns = [col for col in required_columns if col not in df.columns]
if missing_columns:
    print(f"Error: Missing required columns in feature file: {missing_columns}")
    print("Please ensure feature engineering step created all required features.")
    sys.exit(1)

print("Features loaded successfully.")
print("\nDataset Information:")
df.info()

# --- Step 2: Define Features, Targets, and Split Data ---
print("\nStep 2: Preparing data for training...")

# 'X' is our feature set. We drop the two target columns.
X = df.drop(columns=['london_direction', 'london_return'])

# We have two separate targets we want to predict.
y_class = df['london_direction']  # For our classification model
y_reg = df['london_return']      # For our regression model

# --- The Time-Series Split ---
# We will use the first 80% of the data for training and the last 20% for testing.
# This ensures we are always testing on data that comes after our training data.
train_size = int(len(df) * 0.8)

X_train, X_test = X.iloc[:train_size], X.iloc[train_size:]
y_train_class, y_test_class = y_class.iloc[:train_size], y_class.iloc[train_size:]
y_train_reg, y_test_reg = y_reg.iloc[:train_size], y_reg.iloc[train_size:]

print(f"Data split into training and testing sets:")
print(f"X_train shape: {X_train.shape}")
print(f"X_test shape:  {X_test.shape}")

# --- Step 3: Train Classification Model ---
print("\n--- Training Classification Model (Hypothesis A1) ---")

# Initialize the XGBoost Classifier model with some standard parameters.
# 'objective' tells it to perform binary (two-class) classification.
# 'eval_metric' is the metric used to stop training early if it's not improving.
model_class = xgb.XGBClassifier(
    objective='binary:logistic',
    eval_metric='logloss',
    n_estimators=1000, # Number of decision trees to build.
    learning_rate=0.05,
    max_depth=5,
    use_label_encoder=False,
    random_state=42
)

# Train the model on our training data.
model_class.fit(X_train, y_train_class)

# --- Evaluate the Classification Model ---
print("\n--- Evaluating Classification Model ---")

# Make predictions on the unseen test data.
y_pred_class = model_class.predict(X_test)

# Generate classification report and save it
class_report = classification_report(y_test_class, y_pred_class, target_names=['Bearish (0)', 'Bullish (1)'])
print("\nClassification Report:")
print(class_report)

# Check if a report for this configuration already exists
base_pattern = f'model_results_{hypothesis}_{symbol}_{timeframe}_{year}_*.txt'
existing_reports = [f for f in os.listdir('../reports') if f.startswith(f'model_results_{hypothesis}_{symbol}_{timeframe}_{year}_')]

if existing_reports:
    print(f"\nFound existing reports for this configuration:")
    for report in existing_reports:
        print(f"- {report}")
    print("\nSkipping report generation to avoid duplicates.")
    should_save_report = False
else:
    # No existing report found, prepare to save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = f'../reports/model_results_{hypothesis}_{symbol}_{timeframe}_{year}_{timestamp}.txt'
    should_save_report = True

# We'll collect all results before writing to file (only if we should save)
results = []
results.append("=== Model Training Results ===")
results.append(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
results.append(f"Hypothesis: {hypothesis}")
results.append(f"Year: {year}")
results.append(f"Symbol: {symbol}")
results.append(f"Timeframe: {timeframe}")
results.append(f"\nDataset Information:")
results.append(f"Total samples: {len(df)}")
results.append(f"Training samples: {len(X_train)}")
results.append(f"Testing samples: {len(X_test)}")
results.append(f"\n=== Classification Model Results ===")
results.append("Classification Report:")
results.append(class_report)

# --- Step 4: Train Regression Model ---
print("\n--- Training Regression Model (Hypothesis A2) ---")

# Initialize the XGBoost Regressor model.
# 'objective' tells it to minimize the squared error, which is standard for regression.
model_reg = xgb.XGBRegressor(
    objective='reg:squarederror',
    eval_metric='rmse', # Root Mean Squared Error
    n_estimators=1000,
    learning_rate=0.05,
    max_depth=5,
    random_state=42
)

# Train the model on our training data.
model_reg.fit(X_train, y_train_reg)

# --- Evaluate the Regression Model ---
print("\n--- Evaluating Regression Model ---")

# Make predictions on the unseen test data.
y_pred_reg = model_reg.predict(X_test)

# Calculate and print key metrics.
mae = mean_absolute_error(y_test_reg, y_pred_reg)
r2 = r2_score(y_test_reg, y_pred_reg)

print(f"\nRegression Model Results:")
print(f"Mean Absolute Error (MAE): {mae:.6f}")
print("MAE tells us, on average, how far off our return prediction was in percentage points.")
print(f"R-squared (R2 Score): {r2:.4f}")
print("R2 Score tells us how much of the variance in the returns our model can explain (closer to 1 is better).")


# --- Step 5: Save Trained Models ---
print("\nStep 5: Saving models...")

# Define the paths where the models will be saved.
class_model_path = f'../models/xgb_classifier_{hypothesis}_{symbol}_{timeframe}_{year}_present.joblib'
reg_model_path = f'../models/xgb_regressor_{hypothesis}_{symbol}_{timeframe}_{year}_present.joblib'

# Add regression results and feature importance to our results list
results.append(f"\n=== Regression Model Results ===")
results.append(f"Mean Absolute Error (MAE): {mae:.6f}")
results.append(f"R-squared (R2 Score): {r2:.4f}")
results.append("\nMAE: Average prediction error in percentage points")
results.append("R2: Proportion of variance explained by the model (0 to 1, higher is better)")

# Add feature importance information
results.append("\n=== Feature Importance ===")
feature_importance = pd.DataFrame({
    'Feature': X.columns,
    'Importance': model_reg.feature_importances_
}).sort_values('Importance', ascending=False)

results.append("\nTop Features for Regression Model:")
results.append(feature_importance.to_string())

results.append("\n=== Model File Locations ===")
results.append(f"Classification model: {class_model_path}")
results.append(f"Regression model: {reg_model_path}")

# Write results only if no existing report was found
if should_save_report:
    with open(results_file, 'w') as f:
        f.write('\n'.join(results))
    print(f"\nResults saved to: {results_file}")

# Use joblib to dump the trained model objects into files.
joblib.dump(model_class, class_model_path)
joblib.dump(model_reg, reg_model_path)

print(f"Classification model saved to: {class_model_path}")
print(f"Regression model saved to: {reg_model_path}")


