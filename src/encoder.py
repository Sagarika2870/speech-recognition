import torch
import torch.nn as nn
import torch.optim as optim


class Encoder(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, dropout=0.2, bidirectional=True):
        super(Encoder, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout, bidirectional=bidirectional)

    def forward(self, x):
        hidden_states, _ = self.lstm(x)
        return hidden_states