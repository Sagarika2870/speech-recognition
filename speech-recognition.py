import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import matplotlib.pyplot as plt

from pydub import AudioSegment, effects  

import os

########## DATA PROCESSING ##########

# assign directory
input_dir = "./dataset/recordings/recordings/"
output_dir = "./dataset/treated_recordings/"

for filename in os.listdir(input_dir):
    input_path = os.path.join(input_dir, filename)
    # checking if it is a file
    if os.path.isfile(input_path):
        # extra file name
        file_name = (input_path.split("/"))[4]

        # setup output path
        output_path = output_dir + file_name

        # normalize the audio file
        rawsound = AudioSegment.from_file(input_path, "mp3")  
        normalizedsound = effects.normalize(rawsound)  
        normalizedsound.export(output_path, format="mp3")
