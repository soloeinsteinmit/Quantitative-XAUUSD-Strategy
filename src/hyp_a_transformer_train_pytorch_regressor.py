# src/train_pytorch.py (Modified for Session Classification)
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split # Random split is OK here
import numpy as np

from pytorch_utils import TimeSeriesDataset, TransformerModel # Our custom classes
from utils import generate_report

import argparse

# --- Argument Parser ---
parser = argparse.ArgumentParser(description="Create session sequences")
parser.add_argument('--hypothesis', type=str, default="hyp_a", help="Hypothesis (e.g., 'hyp_a').")
parser.add_argument('--symbol', type=str, default="XAUUSD", help="Financial symbol to download (e.g., 'XAUUSD').")
parser.add_argument('--year', type=int, default="2015", help="Year to start downloading data from.")
parser.add_argument('--timeframe', type=str, default="H1", help="Timeframe for the data (e.g., 'M15', 'H1', 'D1').")
parser.add_argument('--epochs', type=int, default=100, help="Number of epochs (e.g., 100")
args = parser.parse_args()

# --- Settings ---
hyp = args.hypothesis.lower()
symbol = args.symbol.lower()
timeframe = args.timeframe.lower()
year = args.year

# --- Step 1: Load Padded Sequence Data ---
print("Step 1: Loading padded session sequence data...")
X = np.load(f'../data/processed/{hyp}_{symbol}_{timeframe}_{year}_session_sequences_X_padded.npy')
y = np.load(f'../data/processed/{hyp}_{symbol}_{timeframe}_{year}_session_targets_y_reg.npy')

# --- Step 2: Create Datasets and DataLoaders ---
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
train_dataset = TimeSeriesDataset(X_train, y_train)
test_dataset = TimeSeriesDataset(X_test, y_test)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
print("DataLoaders created.")

# --- Step 3: Initialize Transformer Model, Loss, and Optimizer ---
input_size = X.shape[2]    # Num features per hour (5)
hidden_size = 32           # The "embedding dimension". Must be divisible by nhead.
nhead = 4                  # Number of "attention heads". 4 or 8 is common.
nlayers = 2                # Number of transformer layers to stack.
output_size = 1

# Initialize our NEW model
model = TransformerModel(input_size, hidden_size, nhead, nlayers, output_size)

# The loss and optimizer are the same as for the LSTM classifier
criterion = nn.L1Loss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
print("Transformer Model, Loss, and Optimizer initialized for REGRESSION.")

# --- Step 4: The Training Loop ---
print("\nStep 4: Starting training loop...")
num_epochs = args.epochs

for epoch in range(num_epochs):
    model.train()
    for batch_X, batch_y in train_loader:
        outputs = model(batch_X).squeeze()
        loss = criterion(outputs, batch_y.float())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

print("Training finished.")

# --- Step 5: Simple Evaluation for REGRESSION ---
model.eval()
total_mae = 0.0
with torch.inference_mode():
    for batch_X, batch_y in test_loader:
        outputs = model(batch_X).squeeze()
        # CHANGE: We directly calculate the MAE. No sigmoid, no rounding.
        mae = criterion(outputs, batch_y.float())
        total_mae += mae.item() * batch_X.size(0)

avg_mae = total_mae / len(test_dataset)
print(f'\nMean Absolute Error (MAE) on test data: {avg_mae:.6f}')
print("This means our model's return prediction is off by this amount on average.")

# --- Step 6: Save the Model --- (This part is unchanged)
model_path = model_path = f'../models/{hyp}_{symbol}_{timeframe}_{year}_pytorch_transformer_regressor.pth'
torch.save(model.state_dict(), model_path)
print(f"\nClassification model saved to {model_path}")

# --- Save result to file ---
generate_report(
    hypothesis=hyp,
    year=year,
    symbol=symbol,
    timeframe=timeframe,
    full_data=len(X),
    X_train=len(X_train),
    X_test=len(X_test),
    model_type="Transformer Regression",
    model_report=f"""
f'\nMean Absolute Error (MAE) on test data: {avg_mae:.6f}'
This means our model's return prediction is off by this amount on average.
    """,
    model_path=model_path
)