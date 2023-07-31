import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

import primary_model as pm
import audio_processing as ap



if __name__ == "__main__":
    # Load dataset paths and transcriptions
    input_dir = "./dataset/treated_recordings/"

    audio_paths = ap.get_audiopath_list(input_dir)
    transcriptions = '''Please call Stella.  Ask her to bring these things with her from the store:  
    Six spoons of fresh snow peas, five thick slabs of blue cheese, and maybe a snack for her brother Bob.  
    We also need a small plastic snake and a big toy frog for the kids.  She can scoop these things into three red bags, 
    and we will go meet her Wednesday at the train station.'''

    # Hyperparameters
    input_dim = 80  # MFCC or spectrogram feature dimension
    hidden_dim = 256
    output_dim = len(audio_paths)  # Number of unique audio files
    batch_size = 16

    # Create the model
    model = pm.SpeechToTextModel(input_dim, hidden_dim, output_dim)

    # Initialize training dataset and dataloader
    dataset = pm.AccentDataset(audio_paths, transcriptions)
    print(dataset)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Loss function, optimizer, and device
    criterion = nn.CTCLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Move the model and criterion to the device
    model.to(device)
    criterion.to(device)

    # Training loop
    num_epochs = 10
    for epoch in range(num_epochs):
        train_loss = pm.train(model, dataloader, criterion, optimizer, device)
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {train_loss:.4f}")

    # Save the trained model for inference
    torch.save(model.state_dict(), "speech_to_text_model.pt")
