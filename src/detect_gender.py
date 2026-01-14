"""Compatibility wrapper around the gender detection service.

Expose `detect_gender(img) -> (gender, confidence)` and delegate
the implementation to `gender_service.GenderDetector`.
"""

from typing import Tuple, Optional
import cv2

from gender_service import build_default_gender_detector, GenderDetectionResult
from crop import crop_gender, cut2cut_crop


_DEFAULT_GENDER_DETECTOR = None


def _get_default_gender_detector():
    global _DEFAULT_GENDER_DETECTOR
    if _DEFAULT_GENDER_DETECTOR is None:
        _DEFAULT_GENDER_DETECTOR = build_default_gender_detector()
    return _DEFAULT_GENDER_DETECTOR


def detect_gender(img) -> Tuple[Optional[str], float]:
    """Detect gender from a cropped voter block image.

    Returns `(gender, confidence)` where gender is `'M'` or `'F'` or `None`.
    """
    if img is None:
        return None, 0.0

    if getattr(img, "ndim", 2) == 3:
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        img_gray = img

    cropped = crop_gender(img_gray)
    cut = cut2cut_crop(cropped)

    detector = _get_default_gender_detector()
    result: GenderDetectionResult = detector.detect(cut)
    return result.gender, result.confidence


if __name__ == "__main__":
    path = "../data/output/number843.jpg"
    img = cv2.imread(path)
    print(detect_gender(img))