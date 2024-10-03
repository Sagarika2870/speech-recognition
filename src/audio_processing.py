import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import matplotlib.pyplot as plt

import os
import time
import csv

########## AUDIO PROCESSING ##########

import librosa
from pydub import AudioSegment, effects  
import noisereduce as nr
import soundfile as sf

import librosa.display

# assign directory
input_dir = "./dataset/original_recordings/"
reduced_noise_dir = "./dataset/red_noise_recordings/"
output_dir = "./dataset/treated_recordings/"

def get_audiopath_list(input_dir):
    audiopath_list = []
    for filename in os.listdir(input_dir):
        input_path = os.path.join(input_dir, filename)
        # checking if it is a file
        if os.path.isfile(input_path):
            # get file name
            audiopath_list.append(input_path)
    return audiopath_list

def get_native_languages():
    language_set = set()
    # open the input CSV file
    input_file = './dataset/speakers_all_treated.csv'
    with open(input_file, 'r') as file:
        reader = csv.reader(file)

        # iterate row by row in the input CSV file
        for row in reader:
            # extract language from row
            if str(row[4]) not in language_set:
                language_set.add(str(row[4]))

    return list(language_set)

def create_new_csv(file_list):
    # open the input CSV file
    input_file = './dataset/speakers_all.csv'
    with open(input_file, 'r') as file:
        reader = csv.reader(file)

        # create a new CSV file for the desired rows
        output_file = './dataset/speakers_all_treated.csv'
        with open(output_file, 'w', newline='') as output:
            writer = csv.writer(output)

            # iterate row by row in the input CSV file
            for row in reader:
                # copy rows that meet the desired conditions
                if str(row[3]) in file_list :
                    writer.writerow(row)
    
def get_status():
    in_count = 0
    out_count = 0
    file_list = []
    for filename in os.listdir(input_dir):
        in_count = in_count + 1
    for filename in os.listdir(output_dir):
        out_count = out_count + 1
        filename = filename.replace(".mp3", "")
        file_list.append(filename)
    return (in_count, out_count, file_list)

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
    start_time = time.time()
    count = 0

    for filename in os.listdir(input_dir):
        try:
            (total, out_count, file_list) = get_status()
            input_path = os.path.join(input_dir, filename)
            # checking if it is a file
            if os.path.isfile(input_path):
                # get file name
                file_name = (input_path.split("/"))[3]
                
                if debug:
                    count = count + 1
                    print(f"Pre-processing file {count} of {total} (File Name: {file_name}))")

                # setup export paths
                reduced_noise_path = reduced_noise_dir + file_name
                output_path = output_dir + file_name
                output_path = output_path.replace(".mp3", ".wav")

                # load audio file
                audio, sr = librosa.load(input_path, sr=None)

                if (sr != 44100):
                    if debug:
                        print("Skipping file with sampling rate: " + str(sr))
                    continue

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
                normalizedsound.export(output_path, format="wav")

                # plot before and after waveforms for the kikongo1.wav file
                #if file_name == "kikongo1.wav":
                #   audio_normalized, sr2 = librosa.load(output_path, sr=None)
                #   print_waveforms(audio1=audio, audio2=reduced_noise, audio3=audio_normalized, sr=sr)
        except Exception as e:
            if debug:
                print(f"Error occurred for {filename}: {e}")
                continue  # continue to the next iteration of the loop

    # calculate the elapsed time
    result = (time.time() - start_time)
    return result

if __name__ == "__main__":
    file_list = []
    #pre_process_audio(debug=True)
    total, successful, file_list = get_status()
    print(f"{successful}/{total} of audio files were pre-processed successfully")
    create_new_csv(file_list)
    language_list = get_native_languages()