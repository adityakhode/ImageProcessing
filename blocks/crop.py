import cv2
import numpy as np

def crop_age(img):
    h, w = img.shape[:2]

    x1 = int(0.088 * w)
    x2 = int(0.16  * w)
    y1 = int(0.83  * h)
    y2 = int(0.96  * h)

    if x2 <= x1 or y2 <= y1:
        raise ValueError("Invalid crop region")

    return img[y1:y2, x1:x2]

def crop_gender(img):
    h, w = img.shape[:2]

    x1 = int(0.40 * w)
    x2 = int(0.46  * w)
    y1 = int(0.84  * h)
    y2 = int(0.99  * h)

    if x2 <= x1 or y2 <= y1:
        raise ValueError("Invalid crop region")

    return img[y1:y2, x1:x2]

def cut2cut_crop(img: np.ndarray) -> np.ndarray:
    """
    Crops image tightly around black foreground pixels.
    Works for white background + black text/digits.

    Parameters:
        img (np.ndarray): Input OpenCV image (BGR or grayscale)

    Returns:
        np.ndarray: Tightly cropped image
    """

    if img is None or img.size == 0:
        raise ValueError("Empty image provided")

    # Convert to grayscale if needed
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img.copy()

    # Binary: black text -> white foreground
    _, binary = cv2.threshold(
        gray, 0, 255,
        cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )

    # Find all foreground pixel coordinates
    coords = cv2.findNonZero(binary)

    # If no foreground detected, return original
    if coords is None:
        return img.copy()

    # Bounding box of all foreground pixels
    x, y, w, h = cv2.boundingRect(coords)

    # Crop original image (preserve color if present)
    return img[y:y+h, x:x+w]
