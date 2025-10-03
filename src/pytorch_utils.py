import torch
import torch.nn as nn
from torch.utils.data import Dataset
import math

# This is the "official PyTorch way" to handle your data.
# It tells PyTorch how to get one item (a sequence and its target) at a time.
class TimeSeriesDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# This is the blueprint for our model's brain.
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size=1):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        
        # The LSTM layer: this is where the sequence processing and "memory" happens.
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        
        # The Linear layer: a standard neural network layer to make the final prediction.
        self.linear = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # The forward pass defines how data flows through the layers.
        lstm_out, _ = self.lstm(x)
        # We only care about the output of the very last time step
        last_time_step_out = lstm_out[:, -1, :]
        prediction = self.linear(last_time_step_out)
        return prediction
    


# Transformers don't inherently know the order of a sequence.
# This class adds information about the position of each time step (hour) to the input data.
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:x.size(0), :]

# This is the blueprint for our Transformer model's brain.
class TransformerModel(nn.Module):
    def __init__(self, input_size, hidden_size, nhead, nlayers, output_size=1):
        super(TransformerModel, self).__init__()
        self.d_model = hidden_size
        
        # 1. An initial layer to get the input into the right shape
        self.embedding = nn.Linear(input_size, hidden_size)
        
        # 2. The Positional Encoding to add time-step information
        self.pos_encoder = PositionalEncoding(hidden_size)
        
        # 3. The core Transformer Encoder layers. This is where the self-attention happens.
        encoder_layers = nn.TransformerEncoderLayer(d_model=hidden_size, nhead=nhead, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers=nlayers)
        
        # 4. The final layer to make a prediction
        self.decoder = nn.Linear(hidden_size, output_size)

    def forward(self, src):
        src = self.embedding(src) * math.sqrt(self.d_model)
        src = self.pos_encoder(src)
        output = self.transformer_encoder(src)
        # We take the output of the very last time step for our final prediction
        output = output[:, -1, :] 
        output = self.decoder(output)
        return output