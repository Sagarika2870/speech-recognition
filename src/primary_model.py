import torch
import torchaudio
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset
import torchaudio.transforms as transforms
from collections import defaultdict
import matplotlib.pyplot as plt

import torch.nn.functional as F
import encoder 
import decoder 
import attention

# Our speech-to-text model
class SpeechToTextModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(SpeechToTextModel, self).__init__()
        self.rnn = nn.LSTM(input_dim, hidden_dim, bidirectional=True, batch_first=True)
        self.fc = nn.Linear(hidden_dim * 2, output_dim)

    def forward(self, x):
        output, _ = self.rnn(x)
        output = self.fc(output)

        output = F.log_softmax(output, dim=-1)

        return output

# Create our custom dataset for loading audio and transcriptions
class AccentDataset(Dataset):
    def __init__(self, audio_paths, transcriptions):
        self.audio_paths = audio_paths
        self.transcriptions = transcriptions
        self.vocab = defaultdict(lambda: len(self.vocab))
        self.transforms = transforms.MFCC(melkwargs={"n_mels": 10, "center": False},
                                          sample_rate=44100, n_mfcc=10, log_mels=True)  # Using MFCC transformation from torchaudio

    def __len__(self):
        return len(self.audio_paths)

    def __getitem__(self, idx):
        audio_path = self.audio_paths[idx]
        waveform, sample_rate = torchaudio.load(audio_path)
        
        # tokenization of transcription
        numerical_transcription = [self.vocab[token] for token in self.transcriptions.split()]
        numerical_transcription_tensor = torch.tensor(numerical_transcription, dtype=torch.long)
        
        # preprocess the audio data using MFCC transformation
        mfcc_features = self.transforms(waveform)
        # remove dim of 1
        mfcc_features = mfcc_features.squeeze(0)
        # go from (input_size, sequence_length) to (sequence_length, input_size)
        mfcc_features = mfcc_features.transpose(0, 1)

        return mfcc_features, numerical_transcription_tensor

def get_accuracy(model, dataloader):
    correct, total = 0, 0
    print("dataloder", list(dataloader)[0])
    for  audio, labels, sequence_len, label_len in dataloader:
        output = model(audio)
        pred = output.max(1, keepdim=True)[1]
        correct += pred.eq(labels.view_as(pred)).sum().item()
        total += labels.shape[0]
    return correct / total

# plotting
def plot(losses, epochs, train_acc, valid_acc):
    plt.title("Training Curve")
    plt.plot(losses, label="Train")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.show()

    plt.title("Training Curve")
    plt.plot(epochs, train_acc, label="Train")
    plt.plot(epochs, valid_acc, label="Validation")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend(loc='best')
    plt.show()

# Training function
def train(model, dataloader,train_loader, valid_loader, batch_size, num_epochs=5, learning_rate=0.001, debug=False):
    criterion = nn.CTCLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    losses, train_acc, valid_acc = [], [], []
    epochs = []

    # GPU support
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    criterion.to(device)

    for epoch in range(num_epochs):
        for audio, label, sequence_len, label_len in train_loader:
            if debug:
                print(f"Audio Shape: {audio.shape}")
                print(f"Label Shape: {label.shape}")
                #print(f"Sequence Length Shape: {sequence_len.shape}")
                #print(f"Label Length Shape: {label_len.shape}")
            audio, label = audio.to(device), label.to(device)
            optimizer.zero_grad()
            outputs = model(audio)
            print(outputs.shape)
            outputs = outputs.permute(1, 0, 2)
            loss = criterion(outputs, label, sequence_len, label_len)
            loss.backward()
            optimizer.step()
        losses.append(float(loss))

        epochs.append(epoch)

        #get accuracy is not working. Incorrect function?
        train_acc.append(get_accuracy(model, train_loader))
        #valid_acc.append(get_accuracy(model, valid_loader))
        print("Epoch %d; Loss %f; Train Acc %f; Val Acc %f" % (
              epoch+1, loss, train_acc[-1], valid_acc[-1]))
        
        if(epoch +1 ) % 2 == 0:
            torch.save(model.state_dict(), f"model_epoch{epoch +1}")
    plot(losses, epochs, train_acc, valid_acc)
    return

# Split dataset into batches of similar sizes
def custom_collate_fn(batch):
    # separate the elements of the batch into separate lists
    waveforms, transcriptions = zip(*batch)

    # convert list of transcriptions into tensor
    transcriptions = torch.stack(transcriptions)
    
    # get the waveform lengths for packing
    lengths= torch.tensor([item.size(1) for item in waveforms])
    max_len = max(lengths)
    
    max_wave_len = max(len(item) for item in waveforms)
    seq_lengths = tuple([max_wave_len for _ in waveforms])
    label_lengths = tuple([len(item) for item in transcriptions])

    print(seq_lengths)
    print(label_lengths)

    # pad the waveforms with zero to the maximum length in the batch
    padded_waveforms = torch.nn.utils.rnn.pad_sequence(
        [torch.nn.functional.pad(item, (0, max_len - item.size(1))) for item in waveforms],
        batch_first=True
    )

    return padded_waveforms.float(), transcriptions.long(), seq_lengths, label_lengths