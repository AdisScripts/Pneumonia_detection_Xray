import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress info and warning messages
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
import cv2
import os
import numpy as np

IMG_SIZE = 150  # Size to which each image will be resized

# Function to load and preprocess data from folders
def load_data(data_dir):
    images = []
    labels = []
    for label in ["PNEUMONIA", "NORMAL"]:
        path = os.path.join(data_dir, label)
        class_num = 1 if label == "PNEUMONIA" else 0
        for img in os.listdir(path):
            try:
                img_array = cv2.imread(os.path.join(path, img), cv2.IMREAD_GRAYSCALE)
                resized_img = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
                images.append(resized_img)
                labels.append(class_num)
            except Exception as e:
                pass
    return np.array(images), np.array(labels)

# Define and compile a simple CNN model
def create_model():
    model = Sequential([
        Conv2D(32, (3,3), activation='relu', input_shape=(IMG_SIZE, IMG_SIZE, 1)),
        MaxPooling2D(2,2),
        Conv2D(64, (3,3), activation='relu'),
        MaxPooling2D(2,2),
        Flatten(),
        Dense(128, activation='relu'),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

# Load data and preprocess
data_dir = 'data'  # Path to the 'data' folder with subfolders 'PNEUMONIA' and 'NORMAL'
images, labels = load_data(data_dir)
images = images.reshape(-1, IMG_SIZE, IMG_SIZE, 1) / 255.0  # Normalize images

# Initialize model and train
model = create_model()
model.fit(images, labels, epochs=10, validation_split=0.2)

# Save the trained model
model.save("saved_model/pneumonia_detection_model.h5")
print("Model training complete and saved in 'saved_model' folder.")
