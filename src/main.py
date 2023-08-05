import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

import primary_model as pm
import audio_processing as ap

from sklearn.model_selection import train_test_split

if __name__ == "__main__":
    # Load dataset paths and transcriptions
    #input_dir = "./dataset/treated_recordings/"
    input_dir = "./dataset/testing/"

    audio_paths = ap.get_audiopath_list(input_dir)
    len_list = len(audio_paths)
    transcriptions = '''Please call Stella.  Ask her to bring these things with her from the store:  
    Six spoons of fresh snow peas, five thick slabs of blue cheese, and maybe a snack for her brother Bob.  
    We also need a small plastic snake and a big toy frog for the kids.  She can scoop these things into three red bags, 
    and we will go meet her Wednesday at the train station.'''


    # Hyperparameters
    input_dim = 10  # MFCC feature dimension
    hidden_dim = 256
    output_dim = len(audio_paths)  # Number of unique audio files
    batch_size = 12

    # Create the model
    model = pm.SpeechToTextModel(input_dim, hidden_dim, output_dim)

    # Initialize training dataset and dataloader
    dataset = pm.AccentDataset(audio_paths, transcriptions)

    #train_input, train_target, valid_target, valid_input = train_test_split(audio_paths, transcriptions, test_size=0.4, random_state=42)
    train_split = 0.6
    valid_split = 0.2
    test_split =  0.2

    train_size = int(train_split*len_list)
    valid_size =  int(valid_split*len_list)
    train_input = audio_paths[:train_size]
    valid_input = audio_paths[train_size:train_size + valid_size]
    test_input = audio_paths[train_size + valid_size:]
    train_dataset = pm.AccentDataset(train_input, transcriptions)
    valid_dataset = pm.AccentDataset(valid_input,transcriptions)

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=pm.custom_collate_fn)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=pm.custom_collate_fn)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=True, collate_fn=pm.custom_collate_fn)

    # Train the model
    pm.train(model, dataloader,train_loader, valid_loader, transcriptions, batch_size, num_epochs=15, learning_rate=0.001, debug=False)
    #get_accuracy(model, test_loader)

    # Save the trained model for inference
    torch.save(model.state_dict(), "speech_to_text_model.pt")
