import argparse
import json
from pathlib import Path

import numpy as np
import tensorflow as tf

from modules.vision import (
    analyze_image_quality,
    load_class_names,
    prepare_input,
    readable_label,
    read_image,
)


PROJECT_DIR = Path(__file__).resolve().parent
MODEL_DIR = PROJECT_DIR / "models"
FALLBACK_CORN_THRESHOLD = 0.90
FALLBACK_DISEASE_THRESHOLD = 0.60


def _read_config(path):
    if not path.is_file():
        return {}
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"Could not read model configuration {path}: {error}") from error
    if not isinstance(value, dict):
        raise ValueError(f"Model configuration must contain a JSON object: {path}")
    return value


def resolve_model_bundle(use_candidate=False):
    if use_candidate:
        corn_config_path = MODEL_DIR / "corn_candidate_config.json"
        disease_config_path = MODEL_DIR / "disease_candidate_config.json"
        corn_config = _read_config(corn_config_path)
        disease_config = _read_config(disease_config_path)
        if not corn_config or not disease_config:
            raise FileNotFoundError(
                "Candidate models are incomplete. Train both models before selecting "
                "candidate inference."
            )
        corn_model_path = MODEL_DIR / corn_config.get("model", "")
        disease_model_path = MODEL_DIR / disease_config.get("model", "")
        corn_classes_path = MODEL_DIR / corn_config.get("classes", "")
        disease_classes_path = MODEL_DIR / disease_config.get("classes", "")
        corn_threshold = corn_config.get("corn_threshold")
        embedding_reference_path = MODEL_DIR / corn_config.get("embedding_reference", "")
        feature_distance_threshold = corn_config.get("corn_feature_distance_threshold")
        disease_threshold = disease_config.get("confidence_threshold")
        calibrated = True
        bundle_name = "Candidate (not promoted)"
    else:
        corn_config = _read_config(MODEL_DIR / "corn_config.json")
        disease_config = _read_config(MODEL_DIR / "disease_config.json")
        corn_model_path = MODEL_DIR / "corn_detector.keras"
        disease_model_path = MODEL_DIR / "disease_model.keras"
        corn_classes_path = MODEL_DIR / "corn_classes.txt"
        disease_classes_path = PROJECT_DIR / "classes.txt"
        embedding_reference_path = None
        feature_distance_threshold = corn_config.get("corn_feature_distance_threshold")
        corn_threshold = corn_config.get("corn_threshold", FALLBACK_CORN_THRESHOLD)
        disease_threshold = disease_config.get(
            "confidence_threshold", FALLBACK_DISEASE_THRESHOLD
        )
        calibrated = (
            "corn_threshold" in corn_config
            and "confidence_threshold" in disease_config
        )
        bundle_name = "Active"

    if corn_threshold is None or disease_threshold is None:
        raise ValueError("Model configuration is missing a required decision threshold.")
    for name, value in (
        ("corn threshold", corn_threshold),
        ("disease threshold", disease_threshold),
    ):
        if not isinstance(value, (int, float)) or not 0.0 <= value <= 1.0:
            raise ValueError(f"Invalid {name} in model configuration: {value}")
    for model_path in (corn_model_path, disease_model_path):
        if not model_path.is_file():
            raise FileNotFoundError(f"Model not found: {model_path}")
    return {
        "corn_model_path": corn_model_path,
        "disease_model_path": disease_model_path,
        "corn_classes": load_class_names(corn_classes_path),
        "disease_classes": load_class_names(disease_classes_path),
        "corn_threshold": float(corn_threshold),
        "embedding_reference_path": embedding_reference_path,
        "feature_distance_threshold": feature_distance_threshold,
        "disease_threshold": float(disease_threshold),
        "thresholds_calibrated": calibrated,
        "bundle_name": bundle_name,
    }


def load_model(path):
    try:
        return tf.keras.models.load_model(path)
    except Exception as error:
        raise RuntimeError(f"Could not load model {path}: {error}") from error


def load_model_bundle(use_candidate=False):
    bundle = resolve_model_bundle(use_candidate)
    bundle["corn_model"] = load_model(bundle["corn_model_path"])
    bundle["disease_model"] = load_model(bundle["disease_model_path"])
    reference_path = bundle["embedding_reference_path"]
    if reference_path is not None:
        if not reference_path.is_file():
            raise FileNotFoundError(f"Corn feature reference not found: {reference_path}")
        try:
            reference = np.load(reference_path, allow_pickle=False)
        except (OSError, ValueError) as error:
            raise ValueError(f"Could not load corn feature reference {reference_path}: {error}") from error
        if (
            reference.ndim != 2
            or reference.shape[0] == 0
            or not np.all(np.isfinite(reference))
        ):
            raise ValueError(f"Invalid corn feature reference shape or values: {reference_path}")
        embedding_model = tf.keras.Model(
            inputs=bundle["corn_model"].input,
            outputs=bundle["corn_model"].get_layer("embedding").output,
        )
        embedding_size = int(embedding_model.output_shape[-1])
        if reference.shape[1] != embedding_size:
            raise ValueError(
                "Corn feature reference does not match the detector embedding dimension."
            )
        reference_norms = np.linalg.norm(reference, axis=1, keepdims=True)
        bundle["corn_reference_embeddings"] = (
            reference.astype(np.float32) / np.maximum(reference_norms, 1e-12)
        )
        bundle["embedding_model"] = embedding_model
        if (
            not isinstance(bundle["feature_distance_threshold"], (int, float))
            or not 0.0 <= bundle["feature_distance_threshold"] <= 2.0
        ):
            raise ValueError("Candidate config has an invalid corn feature-distance cutoff.")
    else:
        bundle["corn_reference_embeddings"] = None
        bundle["embedding_model"] = None
    bundle["open_set_guard_available"] = bundle["embedding_model"] is not None
    return bundle


def predict_rgb(image_rgb, bundle, top_k=3):
    if top_k < 1:
        raise ValueError("top_k must be at least 1.")
    quality = analyze_image_quality(image_rgb)
    image = prepare_input(image_rgb)

    corn_probabilities = np.asarray(
        bundle["corn_model"].predict(image, verbose=0)
    )
    if corn_probabilities.ndim != 2 or corn_probabilities.shape[0] != 1:
        raise ValueError("Corn model must return one probability vector per image.")
    corn_probabilities = corn_probabilities[0]
    corn_classes = bundle["corn_classes"]
    if len(corn_probabilities) != len(corn_classes):
        raise ValueError("Corn model outputs do not match the saved class mapping.")
    if not np.all(np.isfinite(corn_probabilities)):
        raise ValueError("Corn model returned non-finite output.")

    corn_index = next(
        (
            index for index, name in enumerate(corn_classes)
            if name.casefold() == "corn"
        ),
        None,
    )
    if corn_index is None:
        raise ValueError("Corn class mapping must contain a 'corn' entry.")
    corn_confidence = float(corn_probabilities[corn_index])
    predicted_corn = int(np.argmax(corn_probabilities))
    feature_distance = None
    if bundle["embedding_model"] is not None:
        embedding = np.asarray(
            bundle["embedding_model"].predict(image, verbose=0)[0], dtype=np.float32
        )
        embedding /= max(float(np.linalg.norm(embedding)), 1e-12)
        similarities = bundle["corn_reference_embeddings"] @ embedding
        feature_distance = float(1.0 - np.max(similarities))

    result = {
        "status": "rejected",
        "decision": "Not a corn leaf",
        "message": "This image was rejected by the corn verification model.",
        "corn_confidence": corn_confidence,
        "corn_feature_distance": feature_distance,
        "open_set_guard_available": feature_distance is not None,
        "quality": quality,
        "top_predictions": [],
        "thresholds_calibrated": bundle["thresholds_calibrated"],
        "bundle_name": bundle["bundle_name"],
        "disease_classes": bundle["disease_classes"],
    }
    if predicted_corn != corn_index:
        return result
    if corn_confidence < bundle["corn_threshold"]:
        result.update(
            {
                "status": "uncertain",
                "decision": "Corn leaf could not be verified",
                "message": (
                    "Corn is the leading detector class, but its score is below "
                    "the configured verification threshold."
                ),
            }
        )
        return result
    if (
        feature_distance is not None
        and feature_distance > bundle["feature_distance_threshold"]
    ):
        result.update(
            {
                "status": "uncertain",
                "decision": "Corn leaf could not be verified",
                "message": (
                    "The image is outside the validation-calibrated corn feature "
                    "reference; no disease prediction was made."
                ),
            }
        )
        return result

    disease_probabilities = np.asarray(
        bundle["disease_model"].predict(image, verbose=0)
    )
    if disease_probabilities.ndim != 2 or disease_probabilities.shape[0] != 1:
        raise ValueError("Disease model must return one probability vector per image.")
    disease_probabilities = disease_probabilities[0]
    disease_classes = bundle["disease_classes"]
    if len(disease_probabilities) != len(disease_classes):
        raise ValueError("Disease model outputs do not match its saved class mapping.")
    if not np.all(np.isfinite(disease_probabilities)):
        raise ValueError("Disease model returned non-finite output.")

    order = np.argsort(disease_probabilities)[::-1][: min(top_k, len(disease_classes))]
    top_predictions = [
        {
            "class_name": disease_classes[index],
            "label": readable_label(disease_classes[index]),
            "confidence": float(disease_probabilities[index]),
        }
        for index in order
    ]
    best = top_predictions[0]
    accepted = best["confidence"] >= bundle["disease_threshold"]
    result.update(
        {
            "status": "corn" if accepted else "uncertain",
            "decision": "Corn leaf detected",
            "message": (
                best["label"] if accepted
                else "Disease prediction is uncertain at the configured threshold."
            ),
            "prediction": best["class_name"] if accepted else None,
            "prediction_label": best["label"] if accepted else "Uncertain",
            "confidence": best["confidence"],
            "top_predictions": top_predictions,
        }
    )
    return result


def predict_image(image_path, use_candidate=False):
    image = read_image(image_path)
    return predict_rgb(image, load_model_bundle(use_candidate))


def main():
    parser = argparse.ArgumentParser(
        description="Verify a corn leaf, inspect image quality, and classify its disease."
    )
    parser.add_argument("image", help="Path to a local JPG, JPEG, PNG, or WEBP image")
    parser.add_argument(
        "--candidate",
        action="store_true",
        help="Use separately trained candidate models rather than active models.",
    )
    args = parser.parse_args()
    try:
        result = predict_image(args.image, args.candidate)
    except (FileNotFoundError, OSError, ValueError, RuntimeError) as error:
        parser.exit(1, f"Error: {error}\n")

    print(f"Decision: {result['decision']}")
    print(result["message"])
    print(f"Corn score: {result['corn_confidence']:.3f}")
    print(f"Thresholds calibrated from validation: {result['thresholds_calibrated']}")
    if result["corn_feature_distance"] is not None:
        print(f"Corn feature distance: {result['corn_feature_distance']:.6f}")
    else:
        print("Open-set feature-distance guard: unavailable for this legacy model.")
    if result["top_predictions"]:
        print("Top disease candidates:")
        for prediction in result["top_predictions"]:
            print(f"  {prediction['label']}: {prediction['confidence']:.3f}")
    quality = result["quality"]
    print(
        f"Image quality: {quality['width']}x{quality['height']}, "
        f"brightness {quality['brightness']}, contrast {quality['contrast']}, "
        f"Laplacian variance {quality['blur_score']}"
    )
    for warning in quality["warnings"]:
        print(f"Warning: {warning}")
    if quality["retake_recommended"]:
        print("A retake is recommended; image quality does not automatically reject it.")


if __name__ == "__main__":
    main()
