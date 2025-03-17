'''
Not required to install any libraries.
To run this script, execute the following command:
uv can be installed using pip
"pip install uv"
Then run the script using the following command:
"uv run fft_visualisation.py"
'''

# /// script
# requires-python = ">=3.12, <3.13"
# dependencies = [
#     "numpy",
#     "scipy",
#     "matplotlib",
#     "tqdm"
# ]
# ///

import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
from tqdm import tqdm

# Define the dataset path
DATASET_PATH = r"Training_Data_CAD_2016"
PLOT_PATH = r"Plots_CAD_2016"
FILTERED_DATA_PATH = r"Filtered_Data_CAD_2016"
FILTER_PLOT_PATH = r"Filtered_Plots_CAD_2016"

# List of subfolders to process
subfolders = ["training-e-normal-2016", "training-b-normal-2016", "training-e-abnormal-2016", "training-b-abnormal-2016"]

# Create the output directories if they don't exist
if not os.path.exists(PLOT_PATH):
    os.makedirs(PLOT_PATH)
if not os.path.exists(FILTER_PLOT_PATH):
    os.makedirs(FILTER_PLOT_PATH)

def find_ffts(dataset_path, subfolders, plot_path):
    # Loop through each subfolder
    for subfolder in subfolders:
        folder_path = os.path.join(dataset_path, subfolder)  # Full path of subfolder
            
        # Loop through each .wav file in the subfolder
        for filename in tqdm(os.listdir(folder_path)):
            if filename.endswith(".wav"):  # Process only .wav files
                file_path = os.path.join(folder_path, filename)  # Full file path
                
                # Load the heart sound file
                fs, heart_sound = wavfile.read(file_path)
                
                # Compute FFT
                fft_data = np.abs(np.fft.fft(heart_sound))
                freqs = np.fft.fftfreq(len(heart_sound), 1/fs)
                
                # Plot the FFT
                plt.figure(figsize=(10, 5))
                plt.plot(freqs[:len(freqs)//2], fft_data[:len(freqs)//2])  # Plot only positive frequencies
                plt.title(f"Frequency Spectrum of {filename}")
                plt.xlabel("Frequency (Hz)")
                plt.ylabel("Magnitude")
                plt.grid()
                # plt.show()
                plot_save_path = os.path.join(plot_path, subfolder, f"{filename}_FFT.png")
                plt.savefig(plot_save_path)
                plt.close()

# Find FFTs of all files in the dataset
find_ffts(DATASET_PATH, subfolders, PLOT_PATH)
print("FFT plots saved for all files in the dataset!")
find_ffts(FILTERED_DATA_PATH, subfolders, FILTER_PLOT_PATH)
print("FFT plots saved for all files in the filtered dataset!")