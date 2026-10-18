import argparse
import json
import random
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import tensorflow as tf
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras import layers, models

from modules.dataset_utils import list_images, source_groups, stratified_group_split
from modules.vision import IMG_SIZE


PROJECT_DIR = Path(__file__).resolve().parent
DATASET_DIR = PROJECT_DIR / "corn_detector"
MODEL_DIR = PROJECT_DIR / "models"
OUTPUT_DIR = PROJECT_DIR / "outputs"
BATCH_SIZE = 16


def _read_resized(path_tensor):
    path = path_tensor.decode("utf-8")
    image = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Could not decode detector image: {path}")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA)
    return image.astype(np.float32)


def make_dataset(paths, labels, training, seed, batch_size):
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

    return (
        dataset.map(load, num_parallel_calls=tf.data.AUTOTUNE, deterministic=True)
        .batch(batch_size)
        .prefetch(tf.data.AUTOTUNE)
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
    features = layers.GlobalAveragePooling2D(name="embedding")(features)
    features = layers.Dropout(0.3)(features)
    outputs = layers.Dense(number_of_classes, activation="softmax")(features)
    model = models.Model(inputs, outputs)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model, backbone


def threshold_for_max_false_accept(
    probabilities,
    labels,
    corn_index,
    max_rate,
    feature_distances,
    feature_distance_cutoff,
):
    non_corn = labels != corn_index
    if not np.any(non_corn):
        raise ValueError("Validation split must include non-corn images.")
    predicted = np.argmax(probabilities, axis=1)
    candidates = sorted(
        set([0.0, 1.0] + [float(value) for value in probabilities[non_corn, corn_index]])
    )
    for threshold in candidates:
        accepted = (
            (predicted[non_corn] == corn_index)
            & (probabilities[non_corn, corn_index] >= threshold)
            & (feature_distances[non_corn] <= feature_distance_cutoff)
        )
        false_accept_rate = np.mean(accepted)
        if false_accept_rate <= max_rate:
            return threshold
    return 1.0


def extract_embeddings(model, dataset):
    embedding_model = tf.keras.Model(
        inputs=model.input,
        outputs=model.get_layer("embedding").output,
    )
    embeddings = np.asarray(embedding_model.predict(dataset, verbose=0), dtype=np.float32)
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    return embeddings / np.maximum(norms, 1e-12)


def nearest_reference_distance(query_embeddings, reference_embeddings):
    similarity = query_embeddings @ reference_embeddings.T
    return 1.0 - np.max(similarity, axis=1)


def save_training_plots(history, labels, predicted, class_names):
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
    figure.savefig(OUTPUT_DIR / "corn_detector_training_curves.png", dpi=160)
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(6, 5))
    sns.heatmap(
        confusion_matrix(labels, predicted, labels=np.arange(len(class_names))),
        annot=True,
        fmt="d",
        cmap="Greens",
        xticklabels=class_names,
        yticklabels=class_names,
        ax=axis,
    )
    axis.set(xlabel="Predicted", ylabel="Actual", title="Internal held-out test set")
    figure.tight_layout()
    figure.savefig(OUTPUT_DIR / "corn_detector_confusion_matrix.png", dpi=160)
    plt.close(figure)


def main():
    parser = argparse.ArgumentParser(
        description="Train a source-grouped corn-versus-not-corn MobileNetV2 detector."
    )
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--fine-tune-epochs", type=int, default=5)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--patience", type=int, default=4)
    parser.add_argument("--max-validation-false-accept-rate", type=float, default=0.05)
    parser.add_argument("--max-validation-corn-rejection-rate", type=float, default=0.05)
    parser.add_argument(
        "--overwrite-candidate",
        action="store_true",
        help="Allow replacing existing candidate model and metadata files.",
    )
    args = parser.parse_args()
    if min(args.epochs, args.fine_tune_epochs, args.batch_size, args.patience) < 1:
        parser.error("epochs, fine-tune-epochs, batch-size, and patience must be positive")
    if not 0.0 <= args.max_validation_false_accept_rate < 1.0:
        parser.error("--max-validation-false-accept-rate must be in [0, 1)")
    if not 0.0 <= args.max_validation_corn_rejection_rate < 1.0:
        parser.error("--max-validation-corn-rejection-rate must be in [0, 1)")
    if not DATASET_DIR.is_dir():
        raise FileNotFoundError(f"Detector dataset folder not found: {DATASET_DIR}")

    class_dirs = sorted(path for path in DATASET_DIR.iterdir() if path.is_dir())
    class_names = [path.name for path in class_dirs]
    if {name.casefold() for name in class_names} != {"corn", "not_corn"}:
        raise ValueError(
            "corn_detector must contain exactly the folders 'corn' and 'not_corn'; "
            f"found: {class_names}"
        )
    corn_index = next(index for index, name in enumerate(class_names) if name.casefold() == "corn")
    paths = []
    labels = []
    for label, class_dir in enumerate(class_dirs):
        class_paths = list_images(class_dir)
        if not class_paths:
            raise ValueError(f"No supported images found in {class_dir}")
        paths.extend(class_paths)
        labels.extend([label] * len(class_paths))
    labels = np.asarray(labels, dtype=np.int32)

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    checkpoint = MODEL_DIR / "corn_detector_candidate.keras"
    class_path = MODEL_DIR / "corn_candidate_classes.txt"
    config_path = MODEL_DIR / "corn_candidate_config.json"
    reference_path = MODEL_DIR / "corn_candidate_reference_embeddings.npy"
    existing = [
        path for path in (checkpoint, class_path, config_path, reference_path)
        if path.exists()
    ]
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

    groups = source_groups(paths)
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

    train_data = make_dataset(
        [paths[index] for index in train_indices],
        labels[train_indices],
        True,
        args.seed,
        args.batch_size,
    )
    validation_data = make_dataset(
        [paths[index] for index in validation_indices],
        labels[validation_indices],
        False,
        args.seed,
        args.batch_size,
    )
    test_data = make_dataset(
        [paths[index] for index in test_indices],
        labels[test_indices],
        False,
        args.seed,
        args.batch_size,
    )
    class_weight_values = compute_class_weight(
        class_weight="balanced", classes=np.arange(len(class_names)), y=labels[train_indices]
    )
    class_weights = {i: float(value) for i, value in enumerate(class_weight_values)}

    model, backbone = build_model(len(class_names))
    history_data = {"loss": [], "val_loss": [], "accuracy": [], "val_accuracy": []}
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            checkpoint, monitor="val_loss", mode="min", save_best_only=True, verbose=1
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=args.patience, restore_best_weights=True
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.3,
            patience=max(1, args.patience // 2),
            min_lr=1e-7,
        ),
    ]
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
    train_corn_indices = train_indices[labels[train_indices] == corn_index]
    reference_embeddings = extract_embeddings(
        best_model,
        make_dataset(
            [paths[index] for index in train_corn_indices],
            labels[train_corn_indices],
            False,
            args.seed,
            args.batch_size,
        ),
    )
    validation_embeddings = extract_embeddings(best_model, validation_data)
    validation_distances = nearest_reference_distance(
        validation_embeddings, reference_embeddings
    )
    validation_corn = labels[validation_indices] == corn_index
    distance_cutoff = float(
        np.quantile(
            validation_distances[validation_corn],
            1.0 - args.max_validation_corn_rejection_rate,
        )
    )
    validation_probabilities = best_model.predict(validation_data, verbose=0)
    threshold = threshold_for_max_false_accept(
        validation_probabilities,
        labels[validation_indices],
        corn_index,
        args.max_validation_false_accept_rate,
        validation_distances,
        distance_cutoff,
    )
    test_probabilities = best_model.predict(test_data, verbose=0)
    test_predictions = np.argmax(test_probabilities, axis=1)
    test_accuracy = accuracy_score(labels[test_indices], test_predictions)
    actual_corn = labels[test_indices] == corn_index
    test_embeddings = extract_embeddings(best_model, test_data)
    test_distances = nearest_reference_distance(test_embeddings, reference_embeddings)
    score_only_accepted = (
        (test_predictions == corn_index)
        & (test_probabilities[:, corn_index] >= threshold)
    )
    accepted_corn = (
        score_only_accepted
        & (test_distances <= distance_cutoff)
    )
    print(f"Internal held-out test argmax accuracy: {test_accuracy:.4f}")
    print(f"Calibrated threshold: {threshold:.4f}")
    print(
        "Internal test score-only non-corn false-acceptance rate: "
        f"{np.mean(score_only_accepted[~actual_corn]):.4f}"
    )
    print(
        "Internal test combined-gate non-corn false-acceptance rate: "
        f"{np.mean(accepted_corn[~actual_corn]):.4f}"
    )
    print(
        "Internal test combined-gate corn false-rejection rate: "
        f"{np.mean(~accepted_corn[actual_corn]):.4f}"
    )
    print(f"Corn feature-distance cutoff: {distance_cutoff:.6f}")
    print("Threshold selection uses validation data only, not the held-out test.")

    class_path.write_text("".join(f"{name}\n" for name in class_names), encoding="utf-8")
    np.save(reference_path, reference_embeddings)
    config_path.write_text(
        json.dumps(
            {
                "model": checkpoint.name,
                "classes": class_path.name,
                "embedding_reference": reference_path.name,
                "corn_threshold": threshold,
                "corn_feature_distance_threshold": distance_cutoff,
                "max_validation_false_accept_rate": args.max_validation_false_accept_rate,
                "max_validation_corn_rejection_rate": (
                    args.max_validation_corn_rejection_rate
                ),
                "threshold_selection": (
                    "Binary score cutoff selected on validation after a nearest-reference "
                    "feature-distance gate calibrated on validation corn images."
                ),
                "seed": args.seed,
                "image_size": IMG_SIZE,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    save_training_plots(
        history_data, labels[test_indices], test_predictions, class_names
    )
    print(f"Candidate detector: {checkpoint}")
    print(f"Candidate classes: {class_path}")
    print(f"Training plots: {OUTPUT_DIR}")
    print("This candidate does not replace the active detector.")


if __name__ == "__main__":
    main()
