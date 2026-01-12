import cv2
import numpy as np
from crop import crop_age

STANDARD_SIZE = 64

def preprocess_age_digits(img):
    """
    Input:
        Cropped age image (2 digits, white background, black digits)
    Output:
        Clean, sharp, topology-safe binary image
    """

    # 1️⃣ Grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 2️⃣ Edge-preserving smoothing
    # (better than Gaussian for bold fonts)
    gray = cv2.bilateralFilter(gray, d=7, sigmaColor=50, sigmaSpace=50)

    # 3️⃣ Stable binarization
    _, binary = cv2.threshold(
        gray, 0, 255,
        cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )

    # 4️⃣ Fill holes inside digits (VERY IMPORTANT)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

    # 5️⃣ Remove tiny noise without thinning strokes
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)

    # 6️⃣ Edge sharpening (binary-safe)
    edges = cv2.Canny(binary, 50, 150)
    binary = cv2.bitwise_or(binary, edges)
    binary = 255 -binary
    return binary


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

def resize(img: np.ndarray) -> np.ndarray:
    return cv2.resize(img, (STANDARD_SIZE, STANDARD_SIZE))

def split_by_vertical_transitions(img: np.ndarray):
    """
    Split image based on vertical white → black transitions.
    Assumes:
    - White background
    - Black foreground
    - Image already tightly cropped (cut2cut)

    Returns:
        List of np.ndarray image segments
    """

    if img is None or img.size == 0:
        raise ValueError("Empty image")

    # Convert to grayscale if needed
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img.copy()

    h, w = gray.shape

    # Ensure strict binary: white=255, black=0
    _, binary = cv2.threshold(
        gray, 0, 255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    # Column classification
    col_is_white = np.all(binary == 255, axis=0)
    col_has_black = np.any(binary == 0, axis=0)

    split_positions = []

    i = 0
    while i < w - 1:
        # Detect WHITE → BLACK transition
        if col_is_white[i] and col_has_black[i + 1]:
            white_col = i
            black_col = i + 1
            split_x = (white_col + black_col) // 2
            split_positions.append(split_x)
            i = black_col  # jump forward
        else:
            i += 1

    # If no splits found, return whole image
    if not split_positions:
        return [img]

    # Perform edge-to-edge splits
    parts = []
    prev = 0
    for sx in split_positions:
        parts.append(img[:, prev:sx])
        prev = sx
    parts.append(img[:, prev:w])

    return parts

def split_img(img):
    result = []

    cut = cut2cut_crop(crop_age(img))
    binary = preprocess_age_digits(cut)
    cut = cut2cut_crop(binary)

    parts = split_by_vertical_transitions(cut)

    for part in parts:
        part = cut2cut_crop(part)
        part = resize(part)
        result.append(part)

    return result


if __name__ == "__main__":
    path = "../data/output/number19.jpg"
    img = cv2.imread(path)

    splitted_image_array = split_img(img)

    for image in splitted_image_array:
        image = cut2cut_crop(image)
        resize_img = cv2.resize(image, (64, 64))
        cv2.imshow("Processed Age", resize_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        status = input()
        if status == 'y':
            path = f"../data/output/{input()}.png"
            cv2.imwrite(path, image)
