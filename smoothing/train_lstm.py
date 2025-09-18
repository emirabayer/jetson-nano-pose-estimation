import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import ModelCheckpoint
from sklearn.model_selection import train_test_split
import os

# --- 1. CONFIGURATION ---
INPUT_NUMPY_FILE = 'movenet_raw_keypoints.npy'
SEQUENCE_LENGTH = 10  # How many past frames to look at to predict the next one.
LSTM_UNITS = 64       # Number of neurons in the LSTM layer. Keep this small.
EPOCHS = 50
BATCH_SIZE = 64
OUTPUT_MODEL_NAME = 'lstm_smoother_best.keras'
ONNX_MODEL_NAME = 'lstm_smoother.onnx'

# --- 2. LOAD AND PREPARE DATA ---
print(f"Loading data from {INPUT_NUMPY_FILE}...")
# Data shape is expected to be (total_frames, 17, 2) for 17 keypoints (y,x)
all_keypoints = np.load(INPUT_NUMPY_FILE)

# For the LSTM, we flatten the keypoints for each frame.
# (17, 2) -> (34,)
num_frames = all_keypoints.shape[0]
num_keypoints = all_keypoints.shape[1]
num_coords = all_keypoints.shape[2]
features_per_frame = num_keypoints * num_coords # 17 * 2 = 34
data = all_keypoints.reshape(num_frames, features_per_frame)

print(f"Data reshaped to: {data.shape}")

# Create sequences using a sliding window
X, y = [], []
for i in range(len(data) - SEQUENCE_LENGTH):
    X.append(data[i:(i + SEQUENCE_LENGTH)])
    y.append(data[i + SEQUENCE_LENGTH])

X = np.array(X)
y = np.array(y)

print(f"Created {len(X)} sequences.")
print(f"Shape of X: {X.shape}") # (num_sequences, SEQUENCE_LENGTH, 34)
print(f"Shape of y: {y.shape}")   # (num_sequences, 34)

# Split data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"Training samples: {len(X_train)}, Validation samples: {len(X_val)}")

# --- 3. BUILD THE LSTM MODEL ---
print("Building the LSTM model...")
model = Sequential([
    LSTM(LSTM_UNITS, unroll=True, input_shape=(SEQUENCE_LENGTH, features_per_frame)),
    Dense(features_per_frame)
])

model.compile(optimizer='adam', loss='mean_squared_error')
model.summary()

# --- 4. TRAIN THE MODEL ---
print("Starting model training...")
# Save only the best model based on validation loss
checkpoint = ModelCheckpoint(
    OUTPUT_MODEL_NAME, 
    monitor='val_loss', 
    verbose=1, 
    save_best_only=True, 
    mode='min'
)

history = model.fit(
    X_train, y_train,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    validation_data=(X_val, y_val),
    callbacks=[checkpoint]
)

print("Training complete.")




# --- 5. CONVERT TO ONNX FOR TENSORRT ---
import tf2onnx
import json

print(f"\nConverting the best model to ONNX format for TensorRT...")

# Load the best saved Keras model
model = tf.keras.models.load_model(OUTPUT_MODEL_NAME)

# Define the input signature for the ONNX model
# Shape: (batch_size, sequence_length, features)
# We use 'None' for batch_size to allow for dynamic batch sizes, typically 1.
input_signature = [tf.TensorSpec([None, SEQUENCE_LENGTH, features_per_frame], tf.float32, name='input')]

# Convert the model
onnx_model, _ = tf2onnx.convert.from_keras(model, input_signature, opset=11)

# Save the ONNX model
with open(ONNX_MODEL_NAME, "wb") as f:
    f.write(onnx_model.SerializeToString())

print(f"✅ ONNX model saved successfully as '{ONNX_MODEL_NAME}'")

# Also, save the configuration for later use during inference
config = {
    'sequence_length': SEQUENCE_LENGTH,
    'features_per_frame': features_per_frame
}
with open('lstm_config.json', 'w') as f:
    json.dump(config, f)
print("✅ LSTM configuration saved to 'lstm_config.json'")