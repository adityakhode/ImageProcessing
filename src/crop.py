
"""src.crop
-----------------

Utilities for cropping regions of interest from images used in
the ImageProcessing project.

This module provides deterministic crops (for age/gender regions)
and a utility to tightly crop around dark foreground pixels on a
light background.

All functions validate inputs, use snake_case names, and return
copies of image slices to avoid surprises from view semantics.
"""


import cv2
import numpy as np
from typing import Tuple

__all__ = ["crop_age", "crop_gender", "cut2cut_crop"]


def _validate_image_and_shape(img: np.ndarray) -> Tuple[int, int]:
    """Validate image-like input and return height, width.

    Raises
    ------
    ValueError
        If `img` is None or empty.
    """

    if img is None:
        raise ValueError("`img` is None")
    if not isinstance(img, np.ndarray):
        raise TypeError("`img` must be a numpy.ndarray")
    if img.size == 0:
        raise ValueError("Empty image provided")

    h, w = img.shape[:2]
    return h, w


def _clamp(value: int, minimum: int, maximum: int) -> int:
    """Clamp integer `value` to the closed interval [minimum, maximum]."""

    return max(minimum, min(value, maximum))


def crop_age(img: np.ndarray) -> np.ndarray:
    """Crop the image to the (approximate) age field region.

    The function computes coordinates relative to the image size
    so the crop adapts to different image resolutions.

    Parameters
    ----------
    img : np.ndarray
        Input image in BGR or grayscale format.

    Returns
    -------
    np.ndarray
        A copy of the cropped region.

    Raises
    ------
    ValueError
        If the computed crop is empty or invalid.
    """

    h, w = _validate_image_and_shape(img)

    x1 = int(0.088 * w)
    x2 = int(0.16 * w)
    y1 = int(0.83 * h)
    y2 = int(0.96 * h)

    # Clamp coordinates to image bounds
    x1 = _clamp(x1, 0, w)
    x2 = _clamp(x2, 0, w)
    y1 = _clamp(y1, 0, h)
    y2 = _clamp(y2, 0, h)

    if x2 <= x1 or y2 <= y1:
        raise ValueError("Calculated age crop region is empty or invalid")

    return img[y1:y2, x1:x2].copy()


def crop_gender(img: np.ndarray) -> np.ndarray:
    """Crop the image to the (approximate) gender field region.

    Coordinates are computed as fractions of the image dimensions.
    """

    h, w = _validate_image_and_shape(img)

    x1 = int(0.40 * w)
    x2 = int(0.46 * w)
    y1 = int(0.84 * h)
    y2 = int(0.99 * h)

    x1 = _clamp(x1, 0, w)
    x2 = _clamp(x2, 0, w)
    y1 = _clamp(y1, 0, h)
    y2 = _clamp(y2, 0, h)

    if x2 <= x1 or y2 <= y1:
        raise ValueError("Calculated gender crop region is empty or invalid")

    return img[y1:y2, x1:x2].copy()


def cut2cut_crop(img: np.ndarray) -> np.ndarray:
    """Tightly crop the image around dark foreground pixels.

    The routine expects a light background with darker foreground
    (e.g. scanned digits or text). It converts the image to grayscale,
    applies Otsu thresholding with inversion, finds the non-zero
    foreground coordinates, and returns the bounding rectangle.

    Parameters
    ----------
    img : np.ndarray
        BGR or grayscale image.

    Returns
    -------
    np.ndarray
        Cropped image containing all detected foreground pixels. If no
        foreground is found, a copy of the original image is returned.
    """

    _ = _validate_image_and_shape(img)

    # Convert to grayscale if needed
    if img.ndim == 3 and img.shape[2] == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img.copy()

    # Invert binary so dark foreground becomes white (foreground)
    _, binary = cv2.threshold(
        gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU
    )

    coords = cv2.findNonZero(binary)
    if coords is None:
        return img.copy()

    x, y, w, h = cv2.boundingRect(coords)
    return img[y : y + h, x : x + w].copy()
