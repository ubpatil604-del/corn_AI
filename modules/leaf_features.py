import cv2
import numpy as np


def analyze_leaf(image_path):

    image = cv2.imread(image_path)

    if image is None:
        return {
            "status": "Image could not be read"
        }

    # ------------------------------------------------
    # IMAGE SIZE
    # ------------------------------------------------

    height, width = image.shape[:2]

    # ------------------------------------------------
    # BRIGHTNESS
    # ------------------------------------------------

    gray = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY
    )

    brightness = float(
        np.mean(gray)
    )

    # ------------------------------------------------
    # GREEN PIXEL ANALYSIS
    # ------------------------------------------------

    hsv = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2HSV
    )

    lower_green = np.array(
        [25, 30, 30]
    )

    upper_green = np.array(
        [95, 255, 255]
    )

    green_mask = cv2.inRange(
        hsv,
        lower_green,
        upper_green
    )

    green_ratio = (
        np.count_nonzero(green_mask)
        /
        green_mask.size
    )

    # ------------------------------------------------
    # TEXTURE
    # ------------------------------------------------

    texture_value = float(
        np.std(gray)
    )

    # ------------------------------------------------
    # RESULT
    # ------------------------------------------------

    return {

        "image_width": width,

        "image_height": height,

        "brightness": round(
            brightness,
            2
        ),

        "green_ratio": round(
            green_ratio * 100,
            2
        ),

        "texture_score": round(
            texture_value,
            2
        )
    }
