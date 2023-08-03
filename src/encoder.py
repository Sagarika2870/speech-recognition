import torch
import torch.nn as nn

class Encoder(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):
        super(Encoder, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)

    def forward(self, data):
        # data: input tensor of shape (batch_size, sequence_length, input_size)
        hidden_states, _ = self.lstm(data)
        return hidden_states