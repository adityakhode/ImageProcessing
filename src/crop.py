
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

__all__ = ["crop_age", "crop_gender", "cut2cut_crop", "crop_epic_id", "crop_name", "crop_relationship", 
           "crop_relationship_name", "crop_sr_no", "crop_ward_id", "crop_house_no"]


def _validate_image_and_shape(img: np.ndarray) -> Tuple[int, int]:
    if img is None:
        raise ValueError("`img` is None")
    if not isinstance(img, np.ndarray):
        raise TypeError("`img` must be a numpy.ndarray")
    if img.size == 0:
        raise ValueError("Empty image provided")

    h, w = img.shape[:2]
    return h, w


def _clamp(value: int, minimum: int, maximum: int) -> int:
    return max(minimum, min(value, maximum))


def cut2cut_crop(img: np.ndarray) -> np.ndarray:
    _ = _validate_image_and_shape(img)

    if img.ndim == 3 and img.shape[2] == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img.copy()

    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)

    coords = cv2.findNonZero(binary)
    if coords is None:
        return img.copy()

    x, y, w, h = cv2.boundingRect(coords)
    return img[y : y + h, x : x + w].copy()


def crop_sr_no(img: np.ndarray) -> np.ndarray:
    h, w = _validate_image_and_shape(img)

    x1 = int(0.00 * w)
    x2 = int(0.15 * w)
    y1 = int(0.05 * h)
    y2 = int(0.18 * h)

    x1 = _clamp(x1, 0, w)
    x2 = _clamp(x2, 0, w)
    y1 = _clamp(y1, 0, h)
    y2 = _clamp(y2, 0, h)

    if x2 <= x1 or y2 <= y1:
        raise ValueError("Calculated gender crop region is empty or invalid")

    return img[y1:y2, x1:x2].copy()


def crop_epic_id(img: np.ndarray) -> np.ndarray:
    h, w = _validate_image_and_shape(img)

    x1 = int(0.35 * w)
    x2 = int(0.65 * w)
    y1 = int(0.05 * h)
    y2 = int(0.20 * h)

    x1 = _clamp(x1, 0, w)
    x2 = _clamp(x2, 0, w)
    y1 = _clamp(y1, 0, h)
    y2 = _clamp(y2, 0, h)

    if x2 <= x1 or y2 <= y1:
        raise ValueError("Calculated gender crop region is empty or invalid")

    return img[y1:y2, x1:x2].copy()


def crop_ward_id(img: np.ndarray) -> np.ndarray:
    h, w = _validate_image_and_shape(img)

    x1 = int(0.68 * w)
    x2 = int(0.90 * w)
    y1 = int(0.05 * h)
    y2 = int(0.20 * h)

    x1 = _clamp(x1, 0, w)
    x2 = _clamp(x2, 0, w)
    y1 = _clamp(y1, 0, h)
    y2 = _clamp(y2, 0, h)

    if x2 <= x1 or y2 <= y1:
        raise ValueError("Calculated gender crop region is empty or invalid")

    return img[y1:y2, x1:x2].copy()


def crop_name(img: np.ndarray) -> np.ndarray:
    h, w = _validate_image_and_shape(img)

    x1 = int(0.24 * w)
    x2 = int(0.70 * w)
    y1 = int(0.20 * h)
    y2 = int(0.40 * h)

    x1 = _clamp(x1, 0, w)
    x2 = _clamp(x2, 0, w)
    y1 = _clamp(y1, 0, h)
    y2 = _clamp(y2, 0, h)

    if x2 <= x1 or y2 <= y1:
        raise ValueError("Calculated gender crop region is empty or invalid")

    return img[y1:y2, x1:x2].copy()


def crop_relationship(img: np.ndarray) -> np.ndarray:
    h, w = _validate_image_and_shape(img)

    x1 = int(0.00 * w)
    x2 = int(0.18 * w)
    y1 = int(0.50 * h)
    y2 = int(0.65 * h)

    x1 = _clamp(x1, 0, w)
    x2 = _clamp(x2, 0, w)
    y1 = _clamp(y1, 0, h)
    y2 = _clamp(y2, 0, h)

    if x2 <= x1 or y2 <= y1:
        raise ValueError("Calculated gender crop region is empty or invalid")

    return img[y1:y2, x1:x2].copy()


def crop_relationship(img: np.ndarray) -> np.ndarray:
    h, w = _validate_image_and_shape(img)

    x1 = int(0.00 * w)
    x2 = int(0.18 * w)
    y1 = int(0.50 * h)
    y2 = int(0.65 * h)

    x1 = _clamp(x1, 0, w)
    x2 = _clamp(x2, 0, w)
    y1 = _clamp(y1, 0, h)
    y2 = _clamp(y2, 0, h)

    if x2 <= x1 or y2 <= y1:
        raise ValueError("Calculated gender crop region is empty or invalid")

    return img[y1:y2, x1:x2].copy()


def crop_relationship_name(img: np.ndarray) -> np.ndarray:
    h, w = _validate_image_and_shape(img)

    x1 = int(0.24 * w)
    x2 = int(0.70 * w)
    y1 = int(0.50 * h)
    y2 = int(0.70 * h)

    x1 = _clamp(x1, 0, w)
    x2 = _clamp(x2, 0, w)
    y1 = _clamp(y1, 0, h)
    y2 = _clamp(y2, 0, h)

    if x2 <= x1 or y2 <= y1:
        raise ValueError("Calculated gender crop region is empty or invalid")

    return img[y1:y2, x1:x2].copy()


def crop_house_no(img: np.ndarray) -> np.ndarray:
    h, w = _validate_image_and_shape(img)

    x1 = int(0.21 * w)
    x2 = int(0.40 * w)
    y1 = int(0.70 * h)
    y2 = int(0.80 * h)

    x1 = _clamp(x1, 0, w)
    x2 = _clamp(x2, 0, w)
    y1 = _clamp(y1, 0, h)
    y2 = _clamp(y2, 0, h)

    if x2 <= x1 or y2 <= y1:
        raise ValueError("Calculated gender crop region is empty or invalid")

    return img[y1:y2, x1:x2].copy()


def crop_age(img: np.ndarray) -> np.ndarray:
    h, w = _validate_image_and_shape(img)

    x1 = int(0.088 * w)
    x2 = int(0.160 * w)
    y1 = int(0.830 * h)
    y2 = int(0.960 * h)

    x1 = _clamp(x1, 0, w)
    x2 = _clamp(x2, 0, w)
    y1 = _clamp(y1, 0, h)
    y2 = _clamp(y2, 0, h)

    if x2 <= x1 or y2 <= y1:
        raise ValueError("Calculated age crop region is empty or invalid")

    return img[y1:y2, x1:x2].copy()


def crop_gender(img: np.ndarray) -> np.ndarray:
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