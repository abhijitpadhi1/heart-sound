'''
Not required to install any libraries.
To run this script, execute the following command:
uv can be installed using pip
"pip install uv"
Then run the script using the following command:
"uv run model_testing.py"
'''


# /// script
# requires-python = ">=3.13"
# dependencies = [
#     "numpy",
#     "librosa",
#     "tensorflow",
#     "matplotlib",
#     "scikit-learn",
#     "tqdm"
# ]
# ///


# 1. Import necessary libraries
import os
import numpy as np
import librosa
import librosa.display
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm

# 2. Define dataset path
DATASET_PATH = 'Training_Data_CAD_2016' 
TEST_DATA_PATH = "Test_Data_2"
FILE_EXTENSION = ".wav"

test_data_files = [f for f in os.listdir(TEST_DATA_PATH) if f.endswith(FILE_EXTENSION)]


# 3. Define subfolder paths for normal and abnormal sounds
NORMAL_FOLDERS = ["training-b-normal-2016", "training-e-normal-2016"]
ABNORMAL_FOLDERS = ["training-b-abnormal-2016", "training-e-abnormal-2016"]

# Defining Params
MAX_PAD_LEN = 100
SAMPLING_RATE = 16000
NO_OF_MFCC = 20
INPUT_SHAPE = (NO_OF_MFCC, 100, 1)

TEST_SIZE = 0.2
RANDOM_STATE = 42

EPOCHS = 20
BATCH_SIZE = 16

# 4. Function to extract MFCC features from audio files
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

# 5. Load dataset and extract features
X, y = [], []

# Function to process folders
def load_data(label, folders):
    for folder in folders:
        folder_path = os.path.join(DATASET_PATH, folder)
        for file in tqdm(os.listdir(folder_path), desc=f"Processing {folder}"):
            file_path = os.path.join(folder_path, file)
            mfcc_features = extract_mfcc(file_path)
            X.append(mfcc_features)
            y.append(label)

# Load normal and abnormal data
load_data("normal", NORMAL_FOLDERS)
load_data("abnormal", ABNORMAL_FOLDERS)

# 6. Convert lists to numpy arrays
X = np.array(X)
y = np.array(y)

# 7. Encode labels (Convert 'normal' & 'abnormal' to numerical values)
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)
y = to_categorical(y)  # Convert to one-hot encoding

# 8. Reshape X to fit CNN input shape
X = X[..., np.newaxis]  # Shape: (samples, 20, 100, 1)

# 9. Split dataset into training (80%), validation (10%), and testing (10%)
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.2, random_state=RANDOM_STATE)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=RANDOM_STATE)

# 10. Define CNN model architecture
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=INPUT_SHAPE),
    BatchNormalization(),
    MaxPooling2D((2, 2)),

    Conv2D(64, (3, 3), activation='relu'),
    BatchNormalization(),
    MaxPooling2D((2, 2)),

    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.3),
    Dense(2, activation='softmax')  # 2 output classes
])

# 11. Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

model.summary()

# 12. Train the model
history = model.fit(X_train, y_train, epochs=EPOCHS, batch_size=BATCH_SIZE, validation_data=(X_val, y_val))

# 13. Evaluate the model on test data
test_loss, test_acc = model.evaluate(X_test, y_test)
print(f"\nTest Accuracy: {test_acc:.2f}")

# 14. Plot training history
plt.figure(figsize=(10, 5))
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.title('Model Training History')
plt.show()

# 15. Save the trained model
model.save("heart_sound_cnn_model.keras")
print("Model saved as 'heart_sound_cnn_model.keras'")

# 16. Load and test the model on a new audio sample
def predict_heart_sound(file_path):
    mfcc_features = extract_mfcc(file_path)
    mfcc_features = mfcc_features[np.newaxis, ..., np.newaxis]  # Reshape for CNN input

    prediction = model.predict(mfcc_features)
    predicted_label = label_encoder.inverse_transform([np.argmax(prediction)])
    
    return predicted_label[0]

# 17. Example usage: Predict on a new sample
tot = 0
ab = 0
for data_file in test_data_files:
    sample_audio = os.path.join(TEST_DATA_PATH, data_file)
    if os.path.exists(sample_audio):
        prediction = predict_heart_sound(sample_audio)
        print(f"Predicted Class: {prediction}")
        if prediction == "abnormal":
            ab += 1
        tot += 1
    else:
        print("No test sample found! Please provide a valid test audio file.")
print(f"Abnormal {ab} from Total {tot} => {(ab/tot)*100}% Acc")

#sample_audio = 'normal.wav'  # Change this to an actual test sample
#if os.path.exists(sample_audio):
#    prediction = predict_heart_sound(sample_audio)
#    print(f"\nPredicted Class: {prediction}")
#else:
#    print("\nNo test sample found! Please provide a valid test audio file.")

