# src/train_model.py
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split # We will explain why NOT to use this
from sklearn.metrics import classification_report, mean_absolute_error, r2_score
import joblib # For saving our trained models

# --- Step 1: Load Feature Data ---
print("Step 1: Loading features...")

try:
    year = 2015
    df = pd.read_parquet(f'../data/processed/hyp_a_features_{year}_present.parquet')
except FileNotFoundError:
    print("Error: The feature file was not found.")
    print("Please run the 'feature_engineering.py' script first.")
    exit()

print("Features loaded successfully.")
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

# Print a report showing key metrics.
# Precision: Of all the "bullish" predictions, how many were correct?
# Recall: Of all the actual bullish days, how many did we correctly identify?
# F1-Score: A combined score of precision and recall.
print(classification_report(y_test_class, y_pred_class, target_names=['Bearish (0)', 'Bullish (1)']))

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

print(f"Mean Absolute Error (MAE): {mae:.6f}")
print("MAE tells us, on average, how far off our return prediction was in percentage points.")
print(f"R-squared (R2 Score): {r2:.4f}")
print("R2 Score tells us how much of the variance in the returns our model can explain (closer to 1 is better).")

# --- Step 5: Save Trained Models ---
print("\nStep 5: Saving models...")

# Define the paths where the models will be saved.
class_model_path = f'../models/xgb_classifier_hyp_a_{year}_present.joblib'
reg_model_path = f'../models/xgb_regressor_hyp_a_{year}_present.joblib'

# Use joblib to dump the trained model objects into files.
joblib.dump(model_class, class_model_path)
joblib.dump(model_reg, reg_model_path)

print(f"Classification model saved to: {class_model_path}")
print(f"Regression model saved to: {reg_model_path}")