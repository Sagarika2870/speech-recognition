import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import matplotlib.pyplot as plt

import os

########## AUDIO PROCESSING ##########

import librosa
from pydub import AudioSegment, effects  
import noisereduce as nr
import soundfile as sf

import librosa.display

def print_waveforms(audio1, audio2, audio3, sr, same_graph=False):
    if same_graph:
        # create a figure with two subplots
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(14, 12))

        # plot the first waveform
        librosa.display.waveshow(audio1, sr=sr, ax=ax1)
        ax1.set(title='Original', xlabel='Time (s)', ylabel='Amplitude')

        # plot the second waveform
        librosa.display.waveshow(audio2, sr=sr, ax=ax2)
        ax2.set(title='Noise Reduction', xlabel='Time (s)', ylabel='Amplitude')

        # plot the first waveform
        librosa.display.waveshow(audio3, sr=sr, ax=ax3)
        ax1.set(title='Noise Reduction + Normalization', xlabel='Time (s)', ylabel='Amplitude')

        # adjust the layout to avoid overlapping
        plt.tight_layout()

        # display the plot
        plt.show()
    else:
        # plot the first waveform
        plt.figure(figsize=(10, 4))
        librosa.display.waveshow(audio1, sr=sr)
        plt.title("Original")
        plt.xlabel("Time (s)")
        plt.ylabel("Amplitude")
        plt.show()

        # plot the second waveform
        plt.figure(figsize=(10, 4))
        librosa.display.waveshow(audio2, sr=sr)
        plt.title("Noise Reduction")
        plt.xlabel("Time (s)")
        plt.ylabel("Amplitude")
        plt.show()

        # plot the third waveform
        plt.figure(figsize=(10, 4))
        librosa.display.waveshow(audio3, sr=sr)
        plt.title("Noise Reduction + Normalization")
        plt.xlabel("Time (s)")
        plt.ylabel("Amplitude")
        plt.show()

def pre_process_audio(debug=False):
    # assign directory
    input_dir = "./dataset/recordings/recordings/"
    reduced_noise_dir = "./dataset/red_noise_recordings/"
    output_dir = "./dataset/treated_recordings/"

    for filename in os.listdir(input_dir):
        input_path = os.path.join(input_dir, filename)
        # checking if it is a file
        if os.path.isfile(input_path):
            # get file name
            file_name = (input_path.split("/"))[4]

            # setup export paths
            reduced_noise_path = reduced_noise_dir + file_name
            output_path = output_dir + file_name

            # load audio file
            audio, sr = librosa.load(input_path, sr=None)

            if debug:
                print("Sampling rate: " + str(sr))
                print("Duration: " + str(librosa.get_duration(y=audio, sr=sr)))

            # noise reduction
            reduced_noise = nr.reduce_noise(y=audio, sr=sr)

            # export audio file, import into audio segment and delete it
            sf.write(reduced_noise_path, reduced_noise, sr)
            audio_segment = AudioSegment.from_mp3(reduced_noise_path)
            os.remove(reduced_noise_path)

            # normalize the audio segment
            normalizedsound = effects.normalize(audio_segment)

            # save audio segment
            normalizedsound.export(output_path, format="mp3")

            # plot before and after waveforms for the kikongo1.mp3 file
            if file_name == "kikongo1.mp3":
                audio_normalized, sr2 = librosa.load(output_path, sr=None)
                print_waveforms(audio1=audio, audio2=reduced_noise, audio3=audio_normalized, sr=sr)
        break

pre_process_audio(debug=True)