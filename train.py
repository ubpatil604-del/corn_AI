import argparse
import json
import random
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import tensorflow as tf
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras import layers, models

from modules.dataset_utils import (
    DISEASE_CLASS_ALIASES,
    canonical_disease_name,
    list_images,
    source_groups,
    stratified_group_split,
    verify_external_sources_are_distinct,
    verify_group_labels,
)
from modules.vision import IMG_SIZE


PROJECT_DIR = Path(__file__).resolve().parent
DATASET_DIR = PROJECT_DIR / "dataset"
MODEL_DIR = PROJECT_DIR / "models"
OUTPUT_DIR = PROJECT_DIR / "outputs"
CLASS_NAMES = ["healthy", "common rust", "gray leaf spot", "northern leaf blight"]
BATCH_SIZE = 16


def scan_dataset():
    class_dirs = sorted(path for path in DATASET_DIR.iterdir() if path.is_dir())
    if not class_dirs:
        raise ValueError(f"No class folders found in {DATASET_DIR}")

    mapped = {name: [] for name in CLASS_NAMES}
    unexpected = []
    for class_dir in class_dirs:
        canonical = canonical_disease_name(class_dir.name)
        if canonical is None:
            unexpected.append(class_dir.name)
            continue
        paths = list_images(class_dir)
        if not paths:
            raise ValueError(f"No supported images found in {class_dir}")
        mapped[canonical].extend(paths)

    missing = [name for name, paths in mapped.items() if not paths]
    if unexpected or missing:
        details = []
        if missing:
            details.append(f"missing required disease folders/classes: {missing}")
        if unexpected:
            details.append(f"unrecognized folders: {unexpected}")
        raise ValueError(
            "Dataset must contain exactly the four disease classes (folder aliases are "
            f"accepted). {'; '.join(details)}. Supported folder aliases: "
            f"{ {name: sorted(aliases) for name, aliases in DISEASE_CLASS_ALIASES.items()} }"
        )

    paths = []
    labels = []
    for label, class_name in enumerate(CLASS_NAMES):
        paths.extend(mapped[class_name])
        labels.extend([label] * len(mapped[class_name]))
    return paths, np.asarray(labels, dtype=np.int32)


def _read_resized(path_tensor):
    path = path_tensor.decode("utf-8")
    encoded = np.fromfile(path, dtype=np.uint8)
    image = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Could not decode dataset image: {path}")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA)
    return image.astype(np.float32)


def make_dataset(paths, labels, training, seed):
    dataset = tf.data.Dataset.from_tensor_slices(
        ([str(path) for path in paths], np.asarray(labels, dtype=np.int32))
    )
    if training:
        dataset = dataset.shuffle(
            len(paths), seed=seed, reshuffle_each_iteration=True
        )

    def load(path, label):
        image = tf.numpy_function(_read_resized, [path], tf.float32)
        image.set_shape((IMG_SIZE, IMG_SIZE, 3))
        return image, label

    options = tf.data.Options()
    options.experimental_deterministic = True
    return (
        dataset.map(load, num_parallel_calls=tf.data.AUTOTUNE, deterministic=True)
        .batch(BATCH_SIZE)
        .prefetch(tf.data.AUTOTUNE)
        .with_options(options)
    )


def build_model(number_of_classes):
    augmentation = tf.keras.Sequential(
        [
            layers.RandomFlip("horizontal"),
            layers.RandomRotation(0.04),
            layers.RandomTranslation(0.05, 0.05),
            layers.RandomZoom(0.08),
            layers.RandomContrast(0.08),
            layers.RandomBrightness(0.08, value_range=(0, 255)),
        ],
        name="training_augmentation",
    )
    backbone = tf.keras.applications.MobileNetV2(
        input_shape=(IMG_SIZE, IMG_SIZE, 3),
        include_top=False,
        weights="imagenet",
    )
    backbone.trainable = False

    inputs = layers.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
    images = augmentation(inputs)
    images = layers.Rescaling(1.0 / 127.5, offset=-1)(images)
    features = backbone(images, training=False)
    features = layers.GlobalAveragePooling2D()(features)
    features = layers.Dropout(0.3)(features)
    outputs = layers.Dense(number_of_classes, activation="softmax")(features)
    model = models.Model(inputs, outputs)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model, backbone


def make_callbacks(checkpoint, patience):
    return [
        tf.keras.callbacks.ModelCheckpoint(
            checkpoint,
            monitor="val_loss",
            mode="min",
            save_best_only=True,
            verbose=1,
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=patience,
            restore_best_weights=True,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.3,
            patience=max(1, patience // 2),
            min_lr=1e-7,
        ),
    ]


def calibrate_confidence_threshold(probabilities, labels, minimum_precision):
    confidence = np.max(probabilities, axis=1)
    predicted = np.argmax(probabilities, axis=1)
    candidates = sorted(set(float(value) for value in confidence), reverse=True)
    for threshold in candidates:
        accepted = confidence >= threshold
        if np.mean(predicted[accepted] == labels[accepted]) >= minimum_precision:
            return threshold
    return 1.0


def save_training_plots(history, test_labels, test_predictions):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    epochs = range(1, len(history["loss"]) + 1)
    figure, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(epochs, history["accuracy"], label="Train")
    axes[0].plot(epochs, history["val_accuracy"], label="Validation")
    axes[0].set(title="Accuracy", xlabel="Epoch", ylabel="Accuracy")
    axes[0].legend()
    axes[1].plot(epochs, history["loss"], label="Train")
    axes[1].plot(epochs, history["val_loss"], label="Validation")
    axes[1].set(title="Loss", xlabel="Epoch", ylabel="Loss")
    axes[1].legend()
    figure.tight_layout()
    figure.savefig(OUTPUT_DIR / "disease_training_curves.png", dpi=160)
    plt.close(figure)

    matrix = confusion_matrix(
        test_labels, test_predictions, labels=np.arange(len(CLASS_NAMES))
    )
    figure, axis = plt.subplots(figsize=(8, 6))
    sns.heatmap(
        matrix,
        annot=True,
        fmt="d",
        cmap="Greens",
        xticklabels=CLASS_NAMES,
        yticklabels=CLASS_NAMES,
        ax=axis,
    )
    axis.set(xlabel="Predicted", ylabel="Actual", title="Internal held-out test set")
    figure.tight_layout()
    figure.savefig(OUTPUT_DIR / "disease_internal_confusion_matrix.png", dpi=160)
    plt.close(figure)


def main():
    parser = argparse.ArgumentParser(
        description="Train a source-grouped four-class corn disease MobileNetV2 model."
    )
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--fine-tune-epochs", type=int, default=8)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--minimum-validation-precision", type=float, default=0.80)
    parser.add_argument(
        "--overwrite-candidate",
        action="store_true",
        help="Allow replacing existing candidate model and metadata files.",
    )
    args = parser.parse_args()
    if min(args.epochs, args.fine_tune_epochs, args.batch_size, args.patience) < 1:
        parser.error("epochs, fine-tune-epochs, batch-size, and patience must be positive")
    if not 0.0 < args.minimum_validation_precision <= 1.0:
        parser.error("--minimum-validation-precision must be in (0, 1]")
    if not DATASET_DIR.is_dir():
        raise FileNotFoundError(f"Disease dataset folder not found: {DATASET_DIR}")

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    checkpoint = MODEL_DIR / "disease_model_candidate.keras"
    class_path = MODEL_DIR / "disease_candidate_classes.txt"
    config_path = MODEL_DIR / "disease_candidate_config.json"
    artifacts = (checkpoint, class_path, config_path)
    existing = [path for path in artifacts if path.exists()]
    if existing and not args.overwrite_candidate:
        raise FileExistsError(
            f"Candidate artifacts already exist: {existing}. Add --overwrite-candidate "
            "only if you intend to replace them."
        )

    tf.keras.utils.set_random_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    try:
        tf.config.experimental.enable_op_determinism()
    except (AttributeError, RuntimeError):
        pass

    paths, labels = scan_dataset()
    groups = source_groups(paths)
    verify_group_labels(labels, groups, paths)
    test_root = PROJECT_DIR / "test data"
    if test_root.is_dir():
        external_test_paths = [
            path
            for class_dir in test_root.iterdir()
            if class_dir.is_dir()
            for path in list_images(class_dir)
        ]
        verify_external_sources_are_distinct(paths, external_test_paths)
    train_indices, validation_indices, test_indices = stratified_group_split(
        labels, groups, args.seed
    )
    for name, indices in (
        ("Train", train_indices),
        ("Validation", validation_indices),
        ("Internal test", test_indices),
    ):
        print(
            f"{name}: {len(indices)} images, "
            f"{len(set(groups[index] for index in indices))} source groups"
        )
    print("Disease image counts:", {
        class_name: int(np.sum(labels == index))
        for index, class_name in enumerate(CLASS_NAMES)
    })

    train_data = make_dataset(
        [paths[index] for index in train_indices],
        labels[train_indices],
        True,
        args.seed,
    )
    validation_data = make_dataset(
        [paths[index] for index in validation_indices],
        labels[validation_indices],
        False,
        args.seed,
    )
    test_data = make_dataset(
        [paths[index] for index in test_indices],
        labels[test_indices],
        False,
        args.seed,
    )
    class_weights_array = compute_class_weight(
        class_weight="balanced",
        classes=np.arange(len(CLASS_NAMES)),
        y=labels[train_indices],
    )
    class_weights = {
        index: float(weight) for index, weight in enumerate(class_weights_array)
    }

    model, backbone = build_model(len(CLASS_NAMES))
    history_data = {"loss": [], "val_loss": [], "accuracy": [], "val_accuracy": []}
    callbacks = make_callbacks(checkpoint, args.patience)
    first_history = model.fit(
        train_data,
        validation_data=validation_data,
        epochs=args.epochs,
        class_weight=class_weights,
        callbacks=callbacks,
        verbose=1,
    )
    for key in history_data:
        history_data[key].extend(first_history.history.get(key, []))

    if args.fine_tune_epochs:
        backbone.trainable = True
        for layer in backbone.layers[:-30]:
            layer.trainable = False
        for layer in backbone.layers:
            if isinstance(layer, layers.BatchNormalization):
                layer.trainable = False
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"],
        )
        second_history = model.fit(
            train_data,
            validation_data=validation_data,
            initial_epoch=len(history_data["loss"]),
            epochs=len(history_data["loss"]) + args.fine_tune_epochs,
            class_weight=class_weights,
            callbacks=callbacks,
            verbose=1,
        )
        for key in history_data:
            history_data[key].extend(second_history.history.get(key, []))

    best_model = tf.keras.models.load_model(checkpoint)
    validation_probabilities = best_model.predict(validation_data, verbose=0)
    threshold = calibrate_confidence_threshold(
        validation_probabilities,
        labels[validation_indices],
        args.minimum_validation_precision,
    )
    test_probabilities = best_model.predict(test_data, verbose=0)
    test_predictions = np.argmax(test_probabilities, axis=1)
    test_accuracy = accuracy_score(labels[test_indices], test_predictions)
    print(f"Internal held-out test accuracy: {test_accuracy:.4f}")
    print(
        classification_report(
            labels[test_indices],
            test_predictions,
            labels=np.arange(len(CLASS_NAMES)),
            target_names=CLASS_NAMES,
            zero_division=0,
        )
    )
    print(
        f"Validation confidence cutoff for >= {args.minimum_validation_precision:.0%} "
        f"selective precision: {threshold:.4f}"
    )

    (MODEL_DIR / "disease_candidate_classes.txt").write_text(
        "".join(f"{name}\n" for name in CLASS_NAMES), encoding="utf-8"
    )
    (MODEL_DIR / "disease_candidate_config.json").write_text(
        json.dumps(
            {
                "model": checkpoint.name,
                "classes": class_path.name,
                "confidence_threshold": threshold,
                "threshold_selection": (
                    "Highest-coverage validation cutoff meeting the configured "
                    "minimum selective precision; not a calibrated probability."
                ),
                "minimum_validation_precision": args.minimum_validation_precision,
                "seed": args.seed,
                "image_size": IMG_SIZE,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    save_training_plots(history_data, labels[test_indices], test_predictions)
    print(f"Candidate model: {checkpoint}")
    print(f"Candidate classes: {class_path}")
    print(f"Training plots: {OUTPUT_DIR}")
    print("This candidate does not replace the active disease model.")


if __name__ == "__main__":
    main()
