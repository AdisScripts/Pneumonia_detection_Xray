# gui_app.py

import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import cv2
import numpy as np
import tensorflow as tf

# Load the trained model
model = tf.keras.models.load_model("saved_model/pneumonia_detection_model.h5")
IMG_SIZE = 150

def predict_pneumonia(image_path):
    img_array = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    resized_img = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE)) / 255.0
    img_reshaped = np.reshape(resized_img, (1, IMG_SIZE, IMG_SIZE, 1))
    prediction = model.predict(img_reshaped)
    return "PNEUMONIA" if prediction[0][0] > 0.5 else "NORMAL"

def upload_image():
    file_path = filedialog.askopenfilename()
    if file_path:
        img = Image.open(file_path)
        img = img.resize((250, 250))
        img_tk = ImageTk.PhotoImage(img)
        label_image.config(image=img_tk)
        label_image.image = img_tk

        # Predict and display result
        result = predict_pneumonia(file_path)
        label_result.config(text=f"Result: {result}", font=("Arial", 14), fg="blue")

# Set up the main GUI window
root = tk.Tk()
root.title("Pneumonia Detection System")

# GUI Elements
label_instruction = tk.Label(root, text="Upload an X-ray image to check for pneumonia", font=("Arial", 16))
label_instruction.pack(pady=20)

button_upload = tk.Button(root, text="Upload X-ray Image", command=upload_image, font=("Arial", 12))
button_upload.pack(pady=10)

label_image = tk.Label(root)
label_image.pack(pady=10)

label_result = tk.Label(root, text="", font=("Arial", 14))
label_result.pack(pady=20)

root.geometry("400x500")
root.mainloop()
