import torch
import torchaudio
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset


# Our speech-to-text model
class SpeechToTextModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(SpeechToTextModel, self).__init__()
        self.rnn = nn.LSTM(input_dim, hidden_dim, bidirectional=True, batch_first=True)
        self.fc = nn.Linear(hidden_dim * 2, output_dim)

    def forward(self, x):
        output, _ = self.rnn(x)
        output = self.fc(output)
        return output

# Create our custom dataset for loading audio and transcriptions
class AccentDataset(Dataset):
    def __init__(self, audio_paths, transcriptions):
        self.audio_paths = audio_paths
        self.transcriptions = transcriptions

    def __len__(self):
        return len(self.audio_paths)

    def __getitem__(self, idx):
        audio_path = self.audio_paths[idx]
        waveform, sample_rate = torchaudio.load(audio_path)
        transcription = self.transcriptions
        return waveform, transcription

# Training function
def train(model, dataloader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0
    for batch_inputs, batch_targets in dataloader:
        batch_inputs, batch_targets = batch_inputs.to(device), batch_targets.to(device)
        optimizer.zero_grad()
        outputs = model(batch_inputs)
        loss = criterion(outputs, batch_targets)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(dataloader)