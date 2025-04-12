import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt


class HybridCNNLSTM(nn.Module):
    def __init__(self, input_size=6, hidden_size=32, num_layers=1, prediction_horizon=6, dropout=0.3):
        super(HybridCNNLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.prediction_horizon = prediction_horizon
        
        # CNN layers
        self.conv1 = nn.Conv1d(in_channels=input_size, out_channels=16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(in_channels=16, out_channels=32, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)
        self.dropout_cnn = nn.Dropout(dropout)
        
        # Compute CNN output size (assuming sequence_length=3)
        cnn_output_size = 32 * (3 // 2)
        
        # LSTM layer
        self.lstm = nn.LSTM(cnn_output_size, hidden_size, num_layers, batch_first=True)
        self.dropout_lstm = nn.Dropout(dropout)
        
        # Fully connected layer
        self.fc = nn.Linear(hidden_size, 2 * prediction_horizon)

    def forward(self, x):
        batch_size = x.size(0)
        
        # CNN expects input as (batch, channels, sequence_length)
        x = x.permute(0, 2, 1)
        
        # CNN layers
        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.pool(x)
        x = self.dropout_cnn(x)
        
        # Reshape for LSTM
        x = x.permute(0, 2, 1)
        
        # LSTM
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        
        # Take the last timestep's output
        out = self.dropout_lstm(out[:, -1, :])
        
        # Fully connected layer
        out = self.fc(out)
        out = out.view(batch_size, self.prediction_horizon, 2)
        return out

