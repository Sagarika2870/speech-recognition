import torch
import torch.nn as nn
import torch.optim as optim
import attention as Attention

class Decoder(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers):
        super(Decoder, self).__init__()
        self.lstm = nn.LSTM(input_size + hidden_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
        self.attention = Attention(hidden_size)

    def forward(self, x, hidden, encoder_outputs):
        context_vector, attention_weights = self.attention(hidden[-1], encoder_outputs)
        x = torch.cat((x, context_vector), dim=2)
        output, hidden = self.lstm(x, hidden)
        output = self.fc(output)
        return output, hidden, attention_weights