"""Compatibility wrapper around the newer age detection service.

This module keeps the legacy function `detect_age(img)` while delegating
the actual work to `age_service.AgeDetector`. The wrapper constructs a
default detector on first use so existing callers do not need to change.
"""

import cv2
from typing import Tuple
from age_service import build_default_detector, AgeDetectionResult


_DEFAULT_DETECTOR = None


def _get_default_detector():
    global _DEFAULT_DETECTOR
    if _DEFAULT_DETECTOR is None:
        _DEFAULT_DETECTOR = build_default_detector()
    return _DEFAULT_DETECTOR


def detect_age(img) -> Tuple[int, float]:
    """Detect age from a cropped voter block image.

    Returns a tuple `(age:int, confidence:float)`. If detection fails,
    `age` may be `None` or raise a ValueError for invalid inputs.
    """
    detector = _get_default_detector()
    result: AgeDetectionResult = detector.detect(img)
    return result.age, result.confidence


if __name__ == "__main__":
    path = "../data/output/number8.jpg"
    img = cv2.imread(path)
    print(detect_age(img))

