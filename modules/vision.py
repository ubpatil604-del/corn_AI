from pathlib import Path

import cv2
import numpy as np


IMG_SIZE = 224
SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}


def decode_image(data, source="image"):
    encoded = np.frombuffer(data, dtype=np.uint8)
    image_bgr = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    if image_bgr is None:
        raise ValueError(f"Could not read a valid image from {source}.")
    return cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)


def read_image(path):
    path = Path(path).expanduser()
    if not path.is_file():
        raise FileNotFoundError(f"Image file not found: {path}")
    if path.suffix.lower() not in SUPPORTED_EXTENSIONS:
        raise ValueError("Unsupported image format. Use JPG, JPEG, PNG, or WEBP.")
    try:
        return decode_image(path.read_bytes(), str(path))
    except OSError as error:
        raise OSError(f"Could not read image file {path}: {error}") from error


def prepare_input(image_rgb):
    resized = cv2.resize(
        image_rgb,
        (IMG_SIZE, IMG_SIZE),
        interpolation=cv2.INTER_AREA if max(image_rgb.shape[:2]) > IMG_SIZE else cv2.INTER_CUBIC,
    )
    return np.asarray(resized, dtype=np.float32)[np.newaxis, ...]


def analyze_image_quality(image_rgb):
    height, width = image_rgb.shape[:2]
    gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
    brightness = float(np.mean(gray))
    contrast = float(np.std(gray))
    blur_score = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    clipped_percent = float(np.mean((gray <= 5) | (gray >= 250)) * 100)
    warnings = []

    if brightness < 45:
        warnings.append("Image is very dark; leaf details may be difficult to assess.")
    elif brightness > 210:
        warnings.append("Image is very bright; some leaf details may be washed out.")
    if contrast < 25:
        warnings.append("Image has low contrast.")
    if blur_score < 80:
        warnings.append("Image may be blurry (low Laplacian variance).")
    if clipped_percent > 15:
        warnings.append("A large portion of the image is near black or clipped white.")
    if width < 224 or height < 224:
        warnings.append("Image resolution is below 224 pixels on at least one side.")

    return {
        "width": width,
        "height": height,
        "brightness": round(brightness, 2),
        "contrast": round(contrast, 2),
        "blur_score": round(blur_score, 2),
        "clipped_percent": round(clipped_percent, 2),
        "warnings": warnings,
        "retake_recommended": bool(
            blur_score < 35 or brightness < 25 or brightness > 230 or clipped_percent > 35
        ),
    }


def readable_label(class_name):
    normalized = " ".join(class_name.casefold().replace("_", " ").replace("-", " ").split())
    labels = {
        "healthy": "Healthy corn leaf",
        "common rust": "Common rust",
        "rust disease": "Common rust",
        "gray leaf spot": "Gray leaf spot",
        "gray leaf disease": "Gray leaf spot",
        "northern leaf blight": "Northern leaf blight",
        "corn": "Corn leaf",
        "not corn": "Not a corn leaf",
        "not corn leaf": "Not a corn leaf",
        "not corn leaves": "Not a corn leaf",
        "not corn leaf images": "Not a corn leaf",
    }
    return labels.get(normalized, class_name.replace("_", " ").replace("-", " ").title())


def load_class_names(path):
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Class mapping not found: {path}")
    names = [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not names:
        raise ValueError(f"Class mapping is empty: {path}")
    return names
