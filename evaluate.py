import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score

from modules.dataset_utils import (
    canonical_disease_name,
    list_images,
    source_groups,
    stratified_group_split,
    verify_external_sources_are_distinct,
)
from modules.vision import analyze_image_quality, prepare_input, read_image, readable_label
from predict import load_model_bundle


PROJECT_DIR = Path(__file__).resolve().parent
DISEASE_TEST_DIR = PROJECT_DIR / "test data"
DISEASE_TRAIN_DIR = PROJECT_DIR / "dataset"
DETECTOR_DATASET_DIR = PROJECT_DIR / "corn_detector"
OUTPUT_DIR = PROJECT_DIR / "outputs"
FALLBACK_CORN_THRESHOLD = 0.90
FALLBACK_DISEASE_THRESHOLD = 0.60


def predict_paths(model, paths, batch_size=16):
    predictions = []
    for start in range(0, len(paths), batch_size):
        batch = np.concatenate(
            [prepare_input(read_image(path)) for path in paths[start:start + batch_size]],
            axis=0,
        )
        predictions.append(np.asarray(model.predict(batch, verbose=0)))
    if not predictions:
        raise ValueError("No images were supplied for prediction.")
    return np.concatenate(predictions, axis=0)


def save_confusion_matrix(matrix, labels, path, title):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    figure, axis = plt.subplots(figsize=(max(7, len(labels)), 6))
    sns.heatmap(
        matrix,
        annot=True,
        fmt="d",
        cmap="Greens",
        xticklabels=labels,
        yticklabels=labels,
        ax=axis,
    )
    axis.set(xlabel="Predicted", ylabel="Actual", title=title)
    figure.tight_layout()
    figure.savefig(path, dpi=160)
    plt.close(figure)


def evaluate_disease(bundle, model, threshold):
    if not DISEASE_TEST_DIR.is_dir():
        raise FileNotFoundError(f"Independent disease test folder not found: {DISEASE_TEST_DIR}")

    test_dirs = sorted(path for path in DISEASE_TEST_DIR.iterdir() if path.is_dir())
    folder_classes = {}
    for folder in test_dirs:
        canonical = canonical_disease_name(folder.name)
        if canonical is None:
            raise ValueError(f"Unrecognized disease test folder: {folder.name}")
        if canonical in folder_classes:
            raise ValueError(f"Duplicate test folders map to disease class {canonical!r}.")
        folder_classes[canonical] = folder

    model_canonical = [canonical_disease_name(name) for name in bundle["disease_classes"]]
    if any(name is None for name in model_canonical):
        raise ValueError(
            f"Model class mapping contains unsupported disease names: "
            f"{bundle['disease_classes']}"
        )
    if set(model_canonical) != set(folder_classes):
        raise ValueError(
            "Independent disease test folders must exactly match the selected model classes. "
            f"Model classes: {model_canonical}; test folders: {sorted(folder_classes)}"
        )

    class_indices = {name: index for index, name in enumerate(model_canonical)}
    paths = []
    labels = []
    for class_name, folder in folder_classes.items():
        class_paths = list_images(folder)
        if not class_paths:
            raise ValueError(f"No supported test images found in {folder}")
        paths.extend(class_paths)
        labels.extend([class_indices[class_name]] * len(class_paths))
    labels = np.asarray(labels, dtype=np.int32)
    training_paths = [
        path
        for class_dir in DISEASE_TRAIN_DIR.iterdir()
        if class_dir.is_dir()
        for path in list_images(class_dir)
    ]
    verify_external_sources_are_distinct(training_paths, paths)
    probabilities = predict_paths(model, paths)
    if probabilities.ndim != 2 or probabilities.shape[1] != len(bundle["disease_classes"]):
        raise ValueError("Disease model output does not match its saved class mapping.")
    predicted = np.argmax(probabilities, axis=1)
    confidence = np.max(probabilities, axis=1)
    accepted = confidence >= threshold
    class_labels = [readable_label(name) for name in bundle["disease_classes"]]
    matrix = confusion_matrix(labels, predicted, labels=np.arange(len(class_labels)))
    report = classification_report(
        labels,
        predicted,
        labels=np.arange(len(class_labels)),
        target_names=class_labels,
        output_dict=True,
        zero_division=0,
    )
    print("\nDisease test folder (provenance is not inferred by the evaluator):")
    print("Verify it was collected independently and was not used for model or threshold selection.")
    print(f"Images: {len(labels)}")
    print(f"Overall accuracy: {accuracy_score(labels, predicted):.4f}")
    print(f"Macro F1: {f1_score(labels, predicted, average='macro', zero_division=0):.4f}")
    print(f"Confidence cutoff: {threshold:.4f}; coverage: {np.mean(accepted):.4f}")
    print(
        "Accuracy among accepted predictions: "
        f"{accuracy_score(labels[accepted], predicted[accepted]):.4f}"
        if np.any(accepted)
        else "Accuracy among accepted predictions: no predictions met the cutoff."
    )
    print("\nPer-class precision, recall, and F1:")
    print(
        classification_report(
            labels,
            predicted,
            labels=np.arange(len(class_labels)),
            target_names=class_labels,
            zero_division=0,
        )
    )

    lighting = {"low": [], "normal": [], "high": []}
    for path, actual, guess in zip(paths, labels, predicted):
        brightness = analyze_image_quality(read_image(path))["brightness"]
        band = "low" if brightness < 80 else "high" if brightness > 170 else "normal"
        lighting[band].append(int(actual == guess))
    lighting_report = {
        band: {
            "images": len(correct),
            "accuracy": float(np.mean(correct)) if correct else None,
        }
        for band, correct in lighting.items()
    }
    print("Disease accuracy by mean-brightness band:", lighting_report)

    save_confusion_matrix(
        matrix,
        class_labels,
        OUTPUT_DIR / "disease_test_confusion_matrix.png",
        "Independent disease test set",
    )
    return {
        "images": len(labels),
        "independence_verified_by_tool": False,
        "source_overlap_check_passed": True,
        "accuracy": float(accuracy_score(labels, predicted)),
        "macro_f1": float(f1_score(labels, predicted, average="macro", zero_division=0)),
        "confidence_threshold": threshold,
        "coverage": float(np.mean(accepted)),
        "per_class": {
            label: {
                metric: float(report[label][metric])
                for metric in ("precision", "recall", "f1-score", "support")
            }
            for label in class_labels
        },
        "accuracy_by_brightness": lighting_report,
        "confusion_matrix": matrix.tolist(),
    }


def detector_test_partition(seed, external_test_dir=None):
    source_dir = Path(external_test_dir) if external_test_dir else DETECTOR_DATASET_DIR
    if not source_dir.is_dir():
        raise FileNotFoundError(f"Detector test folder not found: {source_dir}")
    class_dirs = sorted(path for path in source_dir.iterdir() if path.is_dir())
    class_names = [path.name for path in class_dirs]
    if {name.casefold() for name in class_names} != {"corn", "not_corn"}:
        raise ValueError(
            f"Detector test data must contain exactly 'corn' and 'not_corn' folders: {source_dir}"
        )
    paths = []
    labels = []
    for label, class_dir in enumerate(class_dirs):
        class_paths = list_images(class_dir)
        if not class_paths:
            raise ValueError(f"No supported detector test images found in {class_dir}")
        paths.extend(class_paths)
        labels.extend([label] * len(class_paths))
    labels = np.asarray(labels, dtype=np.int32)
    if external_test_dir:
        training_paths = [
            path
            for class_dir in DETECTOR_DATASET_DIR.iterdir()
            if class_dir.is_dir()
            for path in list_images(class_dir)
        ]
        verify_external_sources_are_distinct(training_paths, paths)
        test_indices = np.arange(len(labels))
    else:
        groups = source_groups(paths)
        _, _, test_indices = stratified_group_split(labels, groups, seed)
    return (
        [paths[index] for index in test_indices],
        labels[test_indices],
        class_names,
    )


def evaluate_detector(bundle, model, threshold, seed, external_test_dir=None):
    paths, labels, class_names = detector_test_partition(seed, external_test_dir)
    if class_names != bundle["corn_classes"]:
        raise ValueError(
            "Detector data folder order does not match saved model classes. "
            f"Dataset: {class_names}; model: {bundle['corn_classes']}"
        )
    probabilities = predict_paths(model, paths)
    if probabilities.ndim != 2 or probabilities.shape[1] != len(class_names):
        raise ValueError("Corn model output does not match its saved class mapping.")
    corn_index = next(
        (index for index, name in enumerate(class_names) if name.casefold() == "corn"),
        None,
    )
    if corn_index is None:
        raise ValueError("Corn class mapping must contain a 'corn' entry.")
    predicted = np.argmax(probabilities, axis=1)
    feature_distances = None
    distance_cutoff = bundle["feature_distance_threshold"]
    score_only_accepted = (
        (predicted == corn_index)
        & (probabilities[:, corn_index] >= threshold)
    )
    if bundle["embedding_model"] is not None:
        embeddings = predict_paths(bundle["embedding_model"], paths)
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        embeddings = embeddings / np.maximum(norms, 1e-12)
        feature_distances = 1.0 - np.max(
            embeddings @ bundle["corn_reference_embeddings"].T, axis=1
        )
        accepted = (
            (predicted == corn_index)
            & (probabilities[:, corn_index] >= threshold)
            & (feature_distances <= distance_cutoff)
        )
    else:
        accepted = score_only_accepted
    actual_corn = labels == corn_index
    false_accept_rate = (
        float(np.mean(accepted[~actual_corn])) if np.any(~actual_corn) else None
    )
    false_reject_rate = (
        float(np.mean(~accepted[actual_corn])) if np.any(actual_corn) else None
    )
    lighting = {"low": [], "normal": [], "high": []}
    for path, actual, guess in zip(paths, labels, predicted):
        brightness = analyze_image_quality(read_image(path))["brightness"]
        band = "low" if brightness < 80 else "high" if brightness > 170 else "normal"
        lighting[band].append(int(actual == guess))
    lighting_report = {
        band: {
            "images": len(correct),
            "accuracy": float(np.mean(correct)) if correct else None,
        }
        for band, correct in lighting.items()
    }
    matrix = confusion_matrix(labels, predicted, labels=np.arange(len(class_names)))
    print("\nSource-grouped detector test split:")
    print(f"Images: {len(labels)}")
    print(f"Argmax accuracy: {accuracy_score(labels, predicted):.4f}")
    print(f"Corn threshold: {threshold:.4f}")
    print(
        "Score-only non-corn false-acceptance rate: "
        f"{np.mean(score_only_accepted[~actual_corn]) if np.any(~actual_corn) else None}"
    )
    print(f"Non-corn false-acceptance rate: {false_accept_rate}")
    print(f"Corn false-rejection rate: {false_reject_rate}")
    print(f"Detector accuracy by mean-brightness band: {lighting_report}")
    if feature_distances is not None:
        print(f"Corn feature-distance cutoff: {distance_cutoff:.6f}")
        print(
            "Non-corn false-acceptance rate after feature-distance guard: "
            f"{false_accept_rate}"
        )
        accepted_corn = actual_corn & accepted
        print(
            "Mean accepted-corn feature distance: "
            f"{np.mean(feature_distances[accepted_corn]):.6f}"
            if np.any(accepted_corn)
            else "Mean accepted-corn feature distance: no corn images accepted."
        )
    save_confusion_matrix(
        matrix,
        [readable_label(name) for name in class_names],
        OUTPUT_DIR / "corn_detector_test_confusion_matrix.png",
        "Source-grouped detector test split",
    )
    return {
        "images": len(labels),
        "accuracy": float(accuracy_score(labels, predicted)),
        "corn_threshold": threshold,
        "corn_feature_distance_threshold": distance_cutoff,
        "feature_distance_guard_available": feature_distances is not None,
        "accuracy_by_brightness": lighting_report,
        "score_only_non_corn_false_acceptance_rate": (
            float(np.mean(score_only_accepted[~actual_corn]))
            if np.any(~actual_corn)
            else None
        ),
        "non_corn_false_acceptance_rate": false_accept_rate,
        "corn_false_rejection_rate": false_reject_rate,
        "source_overlap_check_passed": external_test_dir is not None,
        "independence_verified_by_tool": False,
        "confusion_matrix": matrix.tolist(),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate active or candidate models without tuning on test data."
    )
    parser.add_argument(
        "--candidate",
        action="store_true",
        help="Evaluate candidate models instead of the active models.",
    )
    parser.add_argument("--corn-threshold", type=float)
    parser.add_argument("--disease-threshold", type=float)
    parser.add_argument(
        "--detector-test-dir",
        type=Path,
        help="Independent folder containing corn/ and not_corn/ directories.",
    )
    parser.add_argument("--seed", type=int, default=123)
    args = parser.parse_args()
    for name, value in (
        ("--corn-threshold", args.corn_threshold),
        ("--disease-threshold", args.disease_threshold),
    ):
        if value is not None and not 0.0 <= value <= 1.0:
            parser.error(f"{name} must be between 0 and 1")

    bundle = load_model_bundle(args.candidate)
    corn_threshold = (
        args.corn_threshold
        if args.corn_threshold is not None
        else bundle["corn_threshold"]
    )
    disease_threshold = (
        args.disease_threshold
        if args.disease_threshold is not None
        else bundle["disease_threshold"]
    )
    if not bundle["thresholds_calibrated"]:
        print(
            "Warning: active-model cutoffs use legacy defaults, not validation-calibrated "
            "values. Do not interpret them as measured probabilities."
        )

    corn_model = bundle["corn_model"]
    disease_model = bundle["disease_model"]
    if args.detector_test_dir is not None:
        detector_results = evaluate_detector(
            bundle,
            corn_model,
            corn_threshold,
            args.seed,
            args.detector_test_dir,
        )
    elif args.candidate:
        detector_results = evaluate_detector(
            bundle, corn_model, corn_threshold, args.seed
        )
    else:
        print(
            "\nSkipping active detector metrics: its training split history is unknown. "
            "Supply --detector-test-dir with independent corn/ and not_corn/ folders."
        )
        detector_results = {
            "skipped": True,
            "reason": "No independent detector test directory was provided.",
        }
    results = {
        "model_bundle": bundle["bundle_name"],
        "thresholds_calibrated": bundle["thresholds_calibrated"],
        "detector": detector_results,
        "disease": evaluate_disease(bundle, disease_model, disease_threshold),
    }
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    report_path = OUTPUT_DIR / (
        "candidate_evaluation.json" if args.candidate else "active_evaluation.json"
    )
    report_path.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    print(f"\nEvaluation report saved to {report_path}")
    print("Thresholds were not selected or changed using this test evaluation.")


if __name__ == "__main__":
    main()
