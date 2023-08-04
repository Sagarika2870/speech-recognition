import torch
import torchaudio
import torchtext
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
        #self.encoder = encoder.Encoder(input_dim, hidden_dim, num_layers=4)
        #self.decoder = decoder.Decoder(input_dim,hidden_dim,output_dim,num_layers=1)
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
        # Load the pre-trained GloVe embeddings
        self.glove = torchtext.vocab.GloVe(name='6B', dim=50)

    def __len__(self):
        return len(self.audio_paths)

    def __getitem__(self, idx):
        audio_path = self.audio_paths[idx]
        waveform, sample_rate = torchaudio.load(audio_path)

        # convert transcript into GloVe embeddings
        numerical_transcription_tensor = sum(self.glove[word] for word in split_transcript(self.transcriptions))
        
        # old tokenization of transcription
        #numerical_transcription = [self.vocab[token] for token in self.transcriptions.split()]
        #numerical_transcription_tensor = torch.tensor(numerical_transcription, dtype=torch.long)
        
        # preprocess the audio data using MFCC transformation
        mfcc_features = self.transforms(waveform)
        # remove dim of 1
        mfcc_features = mfcc_features.squeeze(0)
        # go from (input_size, sequence_length) to (sequence_length, input_size)
        mfcc_features = mfcc_features.transpose(0, 1)

        return mfcc_features, numerical_transcription_tensor
    
def split_transcript(text):
    text = text.replace(".", " . ") \
                .replace(",", " , ") \
                .replace("?", " ? ") \
                .replace("!", " ! ") \
                .replace(";", " ; ") \
                .replace(":", " : ") \
                .replace("-", " - ")
    return text.lower().split()

def tensor_to_words():
    return

def get_accuracy(model, device, dataloader):
    model.eval() # set model to evaluation mode
    correct = 0
    total = 0

    with torch.no_grad():
        for  audio, labels, sequence_len, label_len in dataloader:
            audio, labels = audio.to(device), labels.to(device)
            output = model(audio)
            output = output.permute(1, 0, 2)

            print(output.shape)

            # convert logits to predicted labels
            _, predicted = torch.max(output, 2)
            print(f"Shape pred 1: {predicted.shape}")
            predicted = predicted.transpose(1, 0)  # (T, N) -> (N, T)
            print(f"Shape pred 2: {predicted.shape}")
    
    accuracy = correct / total
    return accuracy

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
            outputs = outputs.permute(1, 0, 2)
            loss = criterion(outputs, label, sequence_len, label_len)
            loss.backward()
            torch.cuda.empty_cache()  # Free up GPU memory
            optimizer.step()

        print("--- Epoch done ---")
        losses.append(float(loss))

        epochs.append(epoch)

        #get accuracy is not working. Incorrect function?
        train_acc.append(get_accuracy(model, device, train_loader))
        valid_acc.append(get_accuracy(model, device, valid_loader))
        print("Epoch %d; Loss %f; Train Acc %f; Val Acc %f" % (
              epoch+1, loss, train_acc[-1], valid_acc[-1]))
        
        if(epoch + 1) % 2 == 0:
            torch.cuda.empty_cache()  # Free up GPU memory
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

    # pad the waveforms with zero to the maximum length in the batch
    padded_waveforms = torch.nn.utils.rnn.pad_sequence(
        [torch.nn.functional.pad(item, (0, max_len - item.size(1))) for item in waveforms],
        batch_first=True
    )

    return padded_waveforms.float(), transcriptions.long(), seq_lengths, label_lengths