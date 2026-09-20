import tensorflow as tf
from tensorflow.keras import layers, models
import os

# ============================================================
# CORN AI - COMPLETE TRAINING SYSTEM
# ============================================================

IMG_SIZE = 224
BATCH_SIZE = 16
EPOCHS = 15

DISEASE_DATASET = "dataset"
DISEASE_TEST = "test data"
CORN_DATASET = "corn_detector"
MODEL_FOLDER = "models"

os.makedirs(MODEL_FOLDER, exist_ok=True)

# ============================================================
# COMMON FUNCTION
# ============================================================

def build_model(number_of_classes):

    augmentation = tf.keras.Sequential([
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.10),
        layers.RandomZoom(0.10),
        layers.RandomContrast(0.10)
    ])

    base_model = tf.keras.applications.MobileNetV2(
        input_shape=(IMG_SIZE, IMG_SIZE, 3),
        include_top=False,
        weights="imagenet"
    )

    base_model.trainable = False

    inputs = layers.Input(
        shape=(IMG_SIZE, IMG_SIZE, 3)
    )

    x = augmentation(inputs)

    x = layers.Rescaling(
        1.0 / 127.5,
        offset=-1
    )(x)

    x = base_model(
        x,
        training=False
    )

    x = layers.GlobalAveragePooling2D()(x)

    x = layers.Dropout(0.4)(x)

    outputs = layers.Dense(
        number_of_classes,
        activation="softmax"
    )(x)

    model = models.Model(
        inputs,
        outputs
    )

    model.compile(
        optimizer=tf.keras.optimizers.Adam(
            learning_rate=0.0001
        ),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    return model

# ============================================================
# TRAIN CORN / NOT-CORN DETECTOR
# ============================================================

print("\n======================================")
print("TRAINING CORN / NOT-CORN DETECTOR")
print("======================================")

if not os.path.exists(CORN_DATASET):
    raise FileNotFoundError(
        "corn_detector folder not found!"
    )

corn_train = tf.keras.utils.image_dataset_from_directory(
    CORN_DATASET,
    validation_split=0.2,
    subset="training",
    seed=123,
    image_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE
)

corn_validation = tf.keras.utils.image_dataset_from_directory(
    CORN_DATASET,
    validation_split=0.2,
    subset="validation",
    seed=123,
    image_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    shuffle=False
)

corn_classes = corn_train.class_names

print("\nCorn detector classes:")
print(corn_classes)

corn_model = build_model(
    len(corn_classes)
)

corn_callbacks = [
    tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=4,
        restore_best_weights=True
    ),

    tf.keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.3,
        patience=2,
        min_lr=0.000001
    )
]

corn_model.fit(
    corn_train,
    validation_data=corn_validation,
    epochs=EPOCHS,
    callbacks=corn_callbacks
)

corn_model.save(
    "models/corn_detector.keras"
)

with open(
    "models/corn_classes.txt",
    "w",
    encoding="utf-8"
) as f:

    for name in corn_classes:
        f.write(name + "\n")

print("\nCorn detector saved.")

# ============================================================
# TRAIN DISEASE MODEL
# ============================================================

print("\n======================================")
print("TRAINING CORN DISEASE MODEL")
print("======================================")

if not os.path.exists(DISEASE_DATASET):
    raise FileNotFoundError(
        "dataset folder not found!"
    )

disease_train = tf.keras.utils.image_dataset_from_directory(
    DISEASE_DATASET,
    validation_split=0.2,
    subset="training",
    seed=123,
    image_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    shuffle=True
)

disease_validation = tf.keras.utils.image_dataset_from_directory(
    DISEASE_DATASET,
    validation_split=0.2,
    subset="validation",
    seed=123,
    image_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    shuffle=False
)

disease_test = tf.keras.utils.image_dataset_from_directory(
    DISEASE_TEST,
    image_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    shuffle=False
)

disease_classes = disease_train.class_names

print("\nDisease classes:")
print(disease_classes)

if disease_classes != disease_test.class_names:

    raise ValueError(
        "\nTraining and test classes do not match!\n"
        f"Training: {disease_classes}\n"
        f"Test: {disease_test.class_names}"
    )

with open(
    "classes.txt",
    "w",
    encoding="utf-8"
) as f:

    for name in disease_classes:
        f.write(name + "\n")

disease_model = build_model(
    len(disease_classes)
)

disease_callbacks = [

    tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=4,
        restore_best_weights=True
    ),

    tf.keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.3,
        patience=2,
        min_lr=0.000001
    )
]

# ============================================================
# TRAIN
# ============================================================

disease_model.fit(
    disease_train,
    validation_data=disease_validation,
    epochs=EPOCHS,
    callbacks=disease_callbacks
)

# ============================================================
# VALIDATION ACCURACY
# ============================================================

print("\n======================================")
print("VALIDATION RESULT")
print("======================================")

val_loss, val_accuracy = disease_model.evaluate(
    disease_validation
)

print(
    f"Validation Accuracy: "
    f"{val_accuracy * 100:.2f}%"
)

# ============================================================
# REAL TEST ACCURACY
# ============================================================

print("\n======================================")
print("INDEPENDENT TEST RESULT")
print("======================================")

test_loss, test_accuracy = disease_model.evaluate(
    disease_test
)

print(
    f"REAL TEST ACCURACY: "
    f"{test_accuracy * 100:.2f}%"
)

# ============================================================
# SAVE DISEASE MODEL
# ============================================================

disease_model.save(
    "models/disease_model.keras"
)

print("\n======================================")
print("TRAINING COMPLETE")
print("======================================")

print("\nSaved models:")

print(
    "models/corn_detector.keras"
)

print(
    "models/disease_model.keras"
)

print(
    "classes.txt"
)

print("\nCorn AI training completed.")
