"""Compatibility wrapper around the SR number detection service.

This module keeps the legacy function `extract_numbers(img)` while delegating
the actual work to `sr_no_service.SRNoDetector`. The wrapper constructs a
default detector on first use so existing callers do not need to change.
"""

import cv2
from typing import Optional
from epic_id_service import build_default_detector, EpicIdDetectionResult


__all__ = ["extract_numbers"]


_DEFAULT_DETECTOR = None


def _get_default_detector():
    """Get or create the default SR number detector."""
    global _DEFAULT_DETECTOR
    if _DEFAULT_DETECTOR is None:
        _DEFAULT_DETECTOR = build_default_detector()
    return _DEFAULT_DETECTOR


def extract_epic_id(img) -> Optional[int]:
    """Extract epic id from a voter block image.

    Args:
        img: Input image (BGR format).

    Returns:
        The detected epic id as an integer, or None if detection fails.
    """
    detector = _get_default_detector()
    result: EpicIdDetectionResult = detector.detect(img)
    return result.epic_id


if __name__ == "__main__":
    # Load an image
    test_img = cv2.imread("../data/4.png")

    # Extract epic id from the image
    epic_id = extract_epic_id(test_img)

    # Print the extracted epic id
    print(f"Extracted Epic ID: {epic_id}")