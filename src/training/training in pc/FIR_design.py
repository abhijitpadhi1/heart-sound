'''
Not required to install any libraries.
To run this script, execute the following command:
uv can be installed using pip
"pip install uv"
Then run the script using the following command:
"uv run FIR_design.py"
'''

# /// script
# requires-python = ">=3.12, <3.13"
# dependencies = [
#     "numpy",
#     "matplotlib",
#     "scipy"
# ]
# ///

import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as signal

# Define parameters
fs = 2000        # Sampling rate (Hz)
cutoff = 900     # Cutoff frequency (Hz)
num_taps = 101   # Filter length (higher gives better filtering)

# Design FIR Low-Pass Filter
fir_coeff = signal.firwin(num_taps, cutoff / (fs / 2), window="hamming", pass_zero=True)

# Plot Frequency Response
w, h = signal.freqz(fir_coeff, worN=8000)
plt.plot(w * fs / (2 * np.pi), np.abs(h))
plt.title("FIR Low-Pass Filter (Cutoff = 900 Hz)")
plt.xlabel("Frequency (Hz)")
plt.ylabel("Gain")
plt.grid()
plt.show()