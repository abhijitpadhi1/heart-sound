'''
Not required to install any libraries.
To run this script, execute the following command:
uv can be installed using pip
"pip install uv"
Then run the script using the following command:
"uv run model_testing.py"
'''

# /// script
# requires-python = ">=3.12, <3.13"
# dependencies = [
#     "numpy",
#     "librosa",
#     "tensorflow",
#     "scikit-learn"
# ]
# ///


import os
import librosa
import numpy as np
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical
import tensorflow as tf

TEST_DATA_abnormal  = "Test_Data_2/abnormal"
TEST_DATA_normal  = "Test_Data_2/normal"
FILE_EXTENSION = ".wav"

test_data_abnormal = [f for f in os.listdir(TEST_DATA_abnormal) if f.endswith(FILE_EXTENSION)]
test_data_normal = [f for f in os.listdir(TEST_DATA_normal) if f.endswith(FILE_EXTENSION)]

# Load the model
model = tf.keras.models.load_model('Model_75.keras')

# Defining Params
MAX_PAD_LEN = 100
SAMPLING_RATE = 16000
NO_OF_MFCC = 20

y = np.array(["normal", "abnormal"])

# 7. Encode labels (Convert 'normal' & 'abnormal' to numerical values)
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)
y = to_categorical(y)  # Convert to one-hot encoding

def extract_mfcc(file_path, max_pad_len=MAX_PAD_LEN):
    audio, sample_rate = librosa.load(file_path, sr=SAMPLING_RATE)  # Load audio
    mfcc = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=NO_OF_MFCC)  # Extract 20 MFCCs

    # Pad or truncate to ensure consistent shape
    if mfcc.shape[1] < max_pad_len:
        pad_width = max_pad_len - mfcc.shape[1]
        mfcc = np.pad(mfcc, ((0, 0), (0, pad_width)), mode='constant')
    else:
        mfcc = mfcc[:, :max_pad_len]

    return mfcc

# 16. Load and test the model on a new audio sample
def predict_heart_sound(file_path):
    mfcc_features = extract_mfcc(file_path)
    mfcc_features = mfcc_features[np.newaxis, ..., np.newaxis]  # Reshape for CNN input

    prediction = model.predict(mfcc_features)
    predicted_label = label_encoder.inverse_transform([np.argmax(prediction)])

    return predicted_label[0]

# 17. Example usage: Predict on a new sample
tot_ab, tot_norm,  ab, norm = 0, 0, 0, 0

for data_file in test_data_normal:
    sample_audio = os.path.join(TEST_DATA_normal, data_file)
    if os.path.exists(sample_audio):
        prediction = predict_heart_sound(sample_audio)
        print(f"Predicted Class: {prediction}")
        if prediction == "normal":
            norm += 1
        tot_norm += 1
    else:
        print("No test sample found! Please provide a valid test audio file.")

for data_file in test_data_abnormal:
    sample_audio = os.path.join(TEST_DATA_abnormal, data_file)
    if os.path.exists(sample_audio):
        prediction = predict_heart_sound(sample_audio)
        print(f"Predicted Class: {prediction}")
        if prediction == "abnormal":
            ab += 1
        tot_ab += 1
    else:
        print("No test sample found! Please provide a valid test audio file.")

print(f"Normal {norm} from Total {tot_norm} => {(norm/tot_norm)*100}% Acc")
print(f"Abnormal {ab} from Total {tot_ab} => {(ab/tot_ab)*100}% Acc")
