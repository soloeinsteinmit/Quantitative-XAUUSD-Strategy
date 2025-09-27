# src/train_transformer.py
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from pytorch_utils import TimeSeriesDataset, TransformerModel 
from utils import generate_report
import argparse

# --- Argument Parser ---
parser = argparse.ArgumentParser(description="Create session sequences")
parser.add_argument('--hypothesis', type=str, default="hyp_a", help="Hypothesis (e.g., 'hyp_a').")
parser.add_argument('--symbol', type=str, default="XAUUSD", help="Financial symbol to download (e.g., 'XAUUSD').")
parser.add_argument('--year', type=str, default="2015", help="Year to start downloading data from.")
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
y = np.load(f'../data/processed/{hyp}_{symbol}_{timeframe}_{year}_session_targets_y_class.npy')

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
criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
print("Transformer Model, Loss, and Optimizer initialized.")

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

# --- Step 5: Detailed Evaluation (This is identical to the LSTM script) ---
print("\n--- Evaluating Transformer Model with Detailed Metrics ---")
model.eval()
all_predictions = []
all_labels = []
with torch.inference_mode():
    for batch_X, batch_y in test_loader:
        outputs = model(batch_X).squeeze()
        predicted = torch.round(torch.sigmoid(outputs))
        all_predictions.extend(predicted.cpu().numpy())
        all_labels.extend(batch_y.cpu().numpy())

print("\n--- Classification Report ---")
report = classification_report(all_labels, all_predictions, target_names=['Bearish (0)', 'Bullish (1)'])
print(report)

print("\n--- Confusion Matrix ---")
cm = confusion_matrix(all_labels, all_predictions)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Greens', 
            xticklabels=['Predicted Bearish', 'Predicted Bullish'], 
            yticklabels=['Actual Bearish', 'Actual Bullish'])
plt.ylabel('Actual Label')
plt.xlabel('Predicted Label')
plt.title('Transformer Confusion Matrix')
# plt.show()
plt.savefig(f"../reports/{hyp}_{symbol}_{timeframe}_{year}_pytorch_transformer_classifier.png")

# --- Step 6: Save the Model ---
model_path = f'../models/{hyp}_{symbol}_{timeframe}_{year}_pytorch_transformer_classifier.pth'
torch.save(model.state_dict(), model_path)
print(f"\nTransformer model saved to {model_path}")

# --- Save result to file ---
generate_report(
    hypothesis=hyp,
    year=year,
    symbol=symbol,
    timeframe=timeframe,
    full_data=len(X),
    X_train=len(X_train),
    X_test=len(X_test),
    model_type="Transformer Classification",
    model_report=report,
    model_path=model_path
)