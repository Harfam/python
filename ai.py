import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import DepthwiseConv2D, Flatten, Dense
from PIL import Image, ImageOps
import numpy as np
from os import system, name
import sys
import os
from contextlib import contextmanager

# Function to clear the console
def clear():
    _ = system('cls' if name == 'nt' else 'clear')

# Suppress scientific notation for clarity
np.set_printoptions(suppress=True)

# Suppress TensorFlow logs and warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
tf.get_logger().setLevel('ERROR')

# Load the labels
with open("labels.txt", "r") as f:
    class_names = f.readlines()

# Capture an image
import ecapture  # Adjusted import based on module structure
ecapture.capture(0, "your image", "thing.jpg")

# Load and preprocess the image
image_path = "thing.jpg"
image = Image.open(image_path).convert("RGB")
image = ImageOps.fit(image, (224, 224), Image.LANCZOS)
image_array = np.asarray(image)
normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1
data = np.expand_dims(normalized_image_array, axis=0)

# Define the model structure
model = Sequential([
    DepthwiseConv2D(kernel_size=(3, 3), strides=(1, 1), padding='same', depth_multiplier=1, activation='relu',
                    use_bias=False, input_shape=(224, 224, 3)),
    Flatten(),
    Dense(len(class_names), activation='softmax')
])

# Load weights into the model
try:
    model.load_weights("keras_Model.h5", by_name=True)
except ValueError as e:
    print("Error loading weights:", e)

# Predict class
prediction = model.predict(data)
index = np.argmax(prediction)
class_name = class_names[index].strip()
confidence_score = prediction[0][index]
dots = 0

while confidence_score < 0.9:
    if dots == 0:
        print("loading")
    elif dots == 1:
        print("loading.")
    elif dots == 2:
        print("loading..")
    elif dots == 3:
        print("loading...")
        dots = -1

    # Reinitialize the model and predict again
    model = Sequential([
        DepthwiseConv2D(kernel_size=(3, 3), strides=(1, 1), padding='same', depth_multiplier=1, activation='relu',
                        use_bias=False, input_shape=(224, 224, 3)),
        Flatten(),
        Dense(len(class_names), activation='softmax')
    ])
    prediction = model.predict(data)
    index = np.argmax(prediction)
    class_name = class_names[index].strip()
    confidence_score = prediction[0][index]

    dots += 1
    clear()

print("Confidence Score:", confidence_score)
print("Class:", class_name)
