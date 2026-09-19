import tensorflow as tf
import numpy as np
from tkinter import Tk, filedialog

# Load trained model
model = tf.keras.models.load_model("crop_disease_model.keras")

# Load class names
with open("classes.txt", "r") as f:
    class_names = [line.strip() for line in f.readlines()]

# Open file selection window
root = Tk()
root.withdraw()

image_path = filedialog.askopenfilename(
    title="Select Bengal Gram Leaf Image",
    filetypes=[
        ("Image files", "*.jpg *.jpeg *.png"),
        ("All files", "*.*")
    ]
)

# Check if image selected
if not image_path:
    print("No image selected.")
    exit()

print("\nSelected image:")
print(image_path)

# Image settings
IMG_SIZE = 224

# Load image
image = tf.keras.utils.load_img(
    image_path,
    target_size=(IMG_SIZE, IMG_SIZE)
)

image_array = tf.keras.utils.img_to_array(image)

# Add batch dimension
image_array = np.expand_dims(image_array, axis=0)

# Prediction
prediction = model.predict(image_array, verbose=0)

predicted_index = np.argmax(prediction[0])
predicted_class = class_names[predicted_index]
confidence = prediction[0][predicted_index] * 100

# Display result
print("\n==============================")
print("   CROP DISEASE DETECTION")
print("==============================")
print("Prediction :", predicted_class)
print("Confidence :", round(confidence, 2), "%")
print("==============================")