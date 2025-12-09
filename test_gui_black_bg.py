import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import tensorflow as tf
import numpy as np
import os
# Import the specific preprocessing function for EfficientNet
from tensorflow.keras.applications.efficientnet import preprocess_input as efficientnet_preprocess

# ---------- CONFIG ----------

# Path to the trained model
MODEL_PATH = "models/efficientnetb3_cub_best.keras"
# Required input size for EfficientNetB3
IMAGE_SIZE = (300, 300)

# Load class names from CUB dataset
CLASSES_FILE = r"C:\Users\snas2\Desktop\ImageClassifier\data\classes.txt"
with open(CLASSES_FILE, "r") as f:
    # Read only the class name (after the class ID)
    CLASS_NAMES = [line.strip().split(" ", 1)[1] for line in f]

# ---------- LOAD MODEL ----------

print("[INFO] Loading model...")
try:
    # Ensure custom metrics are loaded correctly (e.g., top_5_accuracy)
    custom_objects = {
        'top_5_accuracy': tf.keras.metrics.TopKCategoricalAccuracy(k=5, name='top_5_accuracy')
    }
    model = tf.keras.models.load_model(MODEL_PATH, custom_objects=custom_objects)
    print("[INFO] Model loaded successfully.")
except Exception as e:
    print(f"[ERROR] Failed to load model: {e}")
    exit()

# ---------- FUNCTIONS ----------

# Function to load an image and perform prediction
def load_image():
    # Open file selection dialog
    file_path = filedialog.askopenfilename()
    if not file_path:
        return

    # 1. Display the image in the GUI
    img = Image.open(file_path).convert("RGB")
    img_display = img.resize((250, 250))
    img_tk = ImageTk.PhotoImage(img_display)
    image_label.config(image=img_tk)
    image_label.image = img_tk

    # 2. Prepare image for the model (CRITICAL FIX: Use the correct preprocessing)
    img = img.resize(IMAGE_SIZE)
    # Convert PIL image to numpy array with float32 datatype
    img_array = np.array(img, dtype=np.float32)

    # Add batch dimension
    img_array = np.expand_dims(img_array, axis=0)
    
    # Apply the correct EfficientNet preprocessing (Centering and Scaling)
    img_array = efficientnet_preprocess(img_array)

    # 3. Perform prediction
    pred = model.predict(img_array)
    
    # 4. Get results
    # Get the index of the highest probability class
    class_idx = np.argmax(pred)
    # Get the class name
    class_name = CLASS_NAMES[class_idx]  
    # Get the confidence score
    confidence = pred[0][class_idx]

    # Update the results label in the GUI
    result_label.config(text=f"Prediction: {class_name}\nConfidence: {confidence:.4f}")

# ---------- GUI ----------

# Setup the main Tkinter window
root = tk.Tk()
root.title("CUB-200 Image Classifier - Black BG")
# Set black background
root.configure(bg="black")
root.geometry("400x500")

# Load Image button
btn = tk.Button(root, text="Load Image", command=load_image, bg="gray", fg="white")
btn.pack(pady=10)

# Label to display the loaded image
image_label = tk.Label(root, bg="black")
image_label.pack(pady=10)

# Label to display the prediction results
result_label = tk.Label(root, text="", bg="black", fg="white", font=("Arial", 12))
result_label.pack(pady=10)

# Run the main Tkinter loop
root.mainloop()