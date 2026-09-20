import tensorflow as tf
import numpy as np
import cv2
import os

from modules.leaf_features import analyze_leaf

# ============================================================
# SETTINGS
# ============================================================

IMG_SIZE = 224

CORN_MODEL = "models/corn_detector.keras"
DISEASE_MODEL = "models/disease_model.keras"

CORN_CLASSES = "models/corn_classes.txt"
DISEASE_CLASSES = "classes.txt"

RESULT_FOLDER = "results"

os.makedirs(
    RESULT_FOLDER,
    exist_ok=True
)

# ============================================================
# LOAD MODELS
# ============================================================

print("\nLoading Corn AI models...")

corn_model = tf.keras.models.load_model(
    CORN_MODEL
)

disease_model = tf.keras.models.load_model(
    DISEASE_MODEL
)

# ============================================================
# LOAD CLASS NAMES
# ============================================================

with open(
    CORN_CLASSES,
    "r",
    encoding="utf-8"
) as f:

    corn_classes = [
        line.strip()
        for line in f
        if line.strip()
    ]

with open(
    DISEASE_CLASSES,
    "r",
    encoding="utf-8"
) as f:

    disease_classes = [
        line.strip()
        for line in f
        if line.strip()
    ]

# ============================================================
# IMAGE PATH
# ============================================================

image_path = input(
    "\nEnter image path: "
).strip().strip('"')

if not os.path.exists(image_path):

    print("\nERROR: Image not found.")

    exit()

# ============================================================
# IMAGE QUALITY
# ============================================================

image = cv2.imread(
    image_path
)

if image is None:

    print(
        "\nERROR: Could not read image."
    )

    exit()

gray = cv2.cvtColor(
    image,
    cv2.COLOR_BGR2GRAY
)

blur_score = cv2.Laplacian(
    gray,
    cv2.CV_64F
).var()

brightness = np.mean(
    gray
)

print("\n======================================")
print("IMAGE QUALITY")
print("======================================")

print(
    f"Sharpness score : {blur_score:.2f}"
)

print(
    f"Brightness      : {brightness:.2f}"
)

if blur_score < 30:

    print(
        "\nWARNING: Image may be blurry."
    )

if brightness < 40:

    print(
        "\nWARNING: Image may be too dark."
    )

elif brightness > 230:

    print(
        "\nWARNING: Image may be too bright."
    )

# ============================================================
# LEAF FEATURES
# ============================================================

print("\n======================================")
print("LEAF FEATURES")
print("======================================")

features = analyze_leaf(
    image_path
)

for key, value in features.items():

    print(
        f"{key}: {value}"
    )

# ============================================================
# CORN / NOT-CORN DETECTION
# ============================================================

img = tf.keras.utils.load_img(
    image_path,
    target_size=(IMG_SIZE, IMG_SIZE)
)

img_array = tf.keras.utils.img_to_array(
    img
)

img_array = tf.expand_dims(
    img_array,
    axis=0
)

corn_prediction = corn_model.predict(
    img_array,
    verbose=0
)[0]

corn_index = np.argmax(
    corn_prediction
)

corn_label = corn_classes[
    corn_index
]

corn_confidence = (
    corn_prediction[corn_index]
    * 100
)

print("\n======================================")
print("CORN DETECTION")
print("======================================")

print(
    f"Result     : {corn_label}"
)

print(
    f"Confidence : {corn_confidence:.2f}%"
)

# ============================================================
# CORN GATE
# ============================================================

if corn_label.lower() == "not_corn":

    print("\n======================================")
    print("FINAL RESULT")
    print("======================================")

    print(
        "NOT CORN"
    )

    print(
        "Disease detection stopped."
    )

    exit()

# ============================================================
# CORN CONFIDENCE PROTECTION
# ============================================================

if corn_confidence < 70:

    print("\n======================================")
    print("FINAL RESULT")
    print("======================================")

    print(
        "UNCERTAIN IMAGE"
    )

    print(
        "Corn detection confidence is too low."
    )

    exit()

# ============================================================
# DISEASE PREDICTION
# ============================================================

disease_prediction = disease_model.predict(
    img_array,
    verbose=0
)[0]

disease_index = np.argmax(
    disease_prediction
)

disease_label = disease_classes[
    disease_index
]

disease_confidence = (
    disease_prediction[disease_index]
    * 100
)

# ============================================================
# TOP 3 RESULTS
# ============================================================

top_indices = np.argsort(
    disease_prediction
)[::-1]

print("\n======================================")
print("DISEASE PREDICTION")
print("======================================")

for index in top_indices:

    print(
        f"{disease_classes[index]} : "
        f"{disease_prediction[index] * 100:.2f}%"
    )

# ============================================================
# CONFIDENCE PROTECTION
# ============================================================

if disease_confidence < 60:

    final_result = "UNCERTAIN"

else:

    final_result = disease_label

# ============================================================
# FINAL RESULT
# ============================================================

print("\n======================================")
print("🌽 CORN AI FINAL RESULT")
print("======================================")

print(
    f"Corn status : {corn_label}"
)

print(
    f"Disease     : {final_result}"
)

print(
    f"Confidence  : {disease_confidence:.2f}%"
)

# ============================================================
# LEAF INFORMATION
# ============================================================

print("\n======================================")
print("LEAF INFORMATION")
print("======================================")

print(
    f"Green area : "
    f"{features.get('green_ratio', 'N/A')}%"
)

print(
    f"Brightness : "
    f"{features.get('brightness', 'N/A')}"
)

print(
    f"Texture    : "
    f"{features.get('texture_score', 'N/A')}"
)

print("\n======================================")
print("ANALYSIS COMPLETE")
print("======================================")