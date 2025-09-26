# src/train_pytorch.py (Modified for Session Classification)
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split # Random split is OK here
import numpy as np
from pytorch_utils import TimeSeriesDataset, LSTMModel # Our custom classes
from utils import generate_report
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
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
# For session-to-session prediction, a random split is acceptable.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

train_dataset = TimeSeriesDataset(X_train, y_train)
test_dataset = TimeSeriesDataset(X_test, y_test)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
print("DataLoaders created.")

# --- Step 3: Initialize Model, Loss, and Optimizer for CLASSIFICATION ---
input_size = X.shape[2]    # Num features per hour (5)
hidden_size = 50
output_size = 1 # We still output one number, but it will be a logit for classification

model = LSTMModel(input_size, hidden_size, output_size)
# USE THE CORRECT LOSS FOR BINARY CLASSIFICATION!
criterion = nn.BCEWithLogitsLoss() # This is the standard for binary classification
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
print("Model, Loss, and Optimizer initialized for Classification.")

# --- Step 4: The Training Loop ---
print("\nStep 4: Starting training loop...")
num_epochs = args.epochs

for epoch in range(num_epochs):
    model.train()
    for batch_X, batch_y in train_loader:
        outputs = model(batch_X).squeeze() # Get model output
        loss = criterion(outputs, batch_y.float()) # Compare with target

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

print("Training finished.")

# --- Step 5: Detailed Evaluation on Test Set ---
print("\n--- Evaluating Model with Detailed Metrics ---")
model.eval() # Set the model to evaluation mode (very important!)

# We need to collect all predictions and true labels from the test set
all_predictions = []
all_labels = []

# torch.no_grad() tells PyTorch we are not training, so it doesn't need to calculate gradients.
# This makes evaluation much faster and uses less memory.
with torch.no_grad():
    for batch_X, batch_y in test_loader:
        # 1. Get the raw model output (logits)
        outputs = model(batch_X).squeeze()
        
        # 2. Convert logits to final predictions (0 or 1)
        predicted = torch.round(torch.sigmoid(outputs))
        
        # 3. Move predictions and labels to the CPU and convert to NumPy arrays
        #    so that scikit-learn can use them.
        all_predictions.extend(predicted.cpu().numpy())
        all_labels.extend(batch_y.cpu().numpy())

# --- 5a. Generate the Classification Report ---
print("\n--- Classification Report ---")
# This report gives you precision, recall, and f1-score for each class.
report = classification_report(all_labels, all_predictions, target_names=['Bearish (0)', 'Bullish (1)'])
print(report)

# --- 5b. Generate and Visualize the Confusion Matrix ---
print("\n--- Confusion Matrix ---")
cm = confusion_matrix(all_labels, all_predictions)

# Use seaborn to create a nice heatmap for the confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['Predicted Bearish', 'Predicted Bullish'], 
            yticklabels=['Actual Bearish', 'Actual Bullish'])
plt.ylabel('Actual Label')
plt.xlabel('Predicted Label')
plt.title('Confusion Matrix')
# plt.show()
plt.savefig(f"../reports/{hyp}_{symbol}_{timeframe}_{year}_pytorch_lstm_classifier.png")

# --- Step 6: Save the Model --- (This part is unchanged)
model_path = f'../models/{hyp}_{symbol}_{timeframe}_{year}_pytorch_lstm_classifier.pth'
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
    model_type="LSTM Classification",
    model_report=report,
    model_path=model_path
)