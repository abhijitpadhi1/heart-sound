'''
Not required to install any libraries.
To run this script, execute the following command:
uv can be installed using pip
"pip install uv"
Then run the script using the following command:
"uv run FIR_application.py"
'''

# /// script
# requires-python = ">=3.12, <3.13"
# dependencies = [
#     "numpy",
#     "scipy",
#     "tqdm"
# ]
# ///

import os
import numpy as np
import scipy.signal as signal
import scipy.io.wavfile as wavfile
from tqdm import tqdm

# Define dataset path
dataset_path = r"Training_Data_CAD_2016"
store_path = r"Filtered_Data_CAD_2016"

# Define the FIR filter
fs = 2000        # Sampling rate (Hz)
cutoff = 900     # Cutoff frequency (Hz)
num_taps = 101   # Filter length
fir_coeff = signal.firwin(num_taps, cutoff / (fs / 2), window="hamming", pass_zero=True)

# Function to filter and save a WAV file
def filter_wav(file_path):
    # Read the WAV file
    fs, heart_sound = wavfile.read(file_path)

    # Ensure the sampling rate is 2000 Hz
    if fs != 2000:
        print(f"Skipping {file_path}: Sampling rate {fs} Hz does not match 2000 Hz.")
        return

    # Apply the FIR filter
    filtered_signal = signal.lfilter(fir_coeff, 1.0, heart_sound)

    # Normalize & convert to int16 (to prevent distortion)
    filtered_signal = np.int16(filtered_signal / np.max(np.abs(filtered_signal)) * 32767)

    # Save the filtered signal with "_filtered" suffix
    # filtered_file_path = file_path.replace(".wav", "_filtered.wav")
    # wavfile.write(filtered_file_path, fs, filtered_signal)
    # print(f"Filtered file saved: {filtered_file_path}")
    return filtered_signal

# Process all WAV files in dataset subfolders
for subfolder in ["training-e-normal-2016", "training-b-normal-2016", "training-e-abnormal-2016", "training-b-abnormal-2016"]:
    subfolder_path = os.path.join(dataset_path, subfolder)

    for filename in tqdm(os.listdir(subfolder_path)):
        if filename.endswith(".wav"):
            file_path = os.path.join(subfolder_path, filename)
            filtered_signal = filter_wav(file_path)
            store_file_path = os.path.join(store_path, subfolder, filename).replace(".wav", "_filtered.wav")
            wavfile.write(store_file_path, fs, filtered_signal)
            # print(f"Filtered file saved: {store_file_path}")

print("Filtering complete for all WAV files!")