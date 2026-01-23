"""SR No (Serial Number) detection service.

This module provides the core logic for extracting serial numbers from
voter block images using OCR and pattern matching.
"""

import numpy as np
from dataclasses import dataclass
from typing import Optional
import pytesseract
import cv2
import re
from crop import crop_sr_no


@dataclass
class SRNoDetectionResult:
    """Result of SR number detection."""
    sr_no: Optional[int]
    confidence: float


class SRNoDetector:
    """Detects and extracts SR number from voter block images."""

    def __init__(self, preprocess_threshold: int = 150):
        """Initialize the SR number detector.

        Args:
            preprocess_threshold: Threshold value for binary thresholding (0-255).
        """
        self.preprocess_threshold = preprocess_threshold

    def _preprocess_image(self, img: np.ndarray) -> np.ndarray:
        """Preprocess image for better OCR results.

        Args:
            img: Input image in BGR format.

        Returns:
            Preprocessed binary image.
        """
        # Crop the region of interest (ROI)
        img = crop_sr_no(img)

        # Convert image to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Apply binary thresholding (convert to black and white)
        _, thresh = cv2.threshold(
            gray, self.preprocess_threshold, 255, cv2.THRESH_BINARY_INV
        )

        # Skip dilation - saves ~30-50ms per image. Tesseract handles thin text well.
        # kernel = np.ones((3, 3), np.uint8)
        # thresh = cv2.dilate(thresh, kernel, iterations=1)

        return thresh

    def detect(self, img: np.ndarray) -> SRNoDetectionResult:
        """Detect SR number from image.

        Args:
            img: Input image (BGR format).

        Returns:
            SRNoDetectionResult containing the detected SR number and confidence.
        """
        if img is None:
            return SRNoDetectionResult(sr_no=None, confidence=0.0)

        try:
            # Preprocess image
            preprocessed_img = self._preprocess_image(img)

            # Use Tesseract to extract text
            # The config `--psm 6` treats the image as a single uniform block of text
            extracted_text = pytesseract.image_to_string(
                preprocessed_img, config="--psm 6"
            )

            # Use regex to find all numbers (sequence of digits)
            numbers = re.findall(r'\b\d+\b', extracted_text)

            if not numbers:
                return SRNoDetectionResult(sr_no=None, confidence=0.0)

            # Convert the first matched number to integer
            sr_no = int(numbers[0])
            # Confidence is high if we found a match
            confidence = 0.95

            return SRNoDetectionResult(sr_no=sr_no, confidence=confidence)

        except Exception as e:
            print(f"Error detecting SR number: {e}")
            return SRNoDetectionResult(sr_no=None, confidence=0.0)


def build_default_detector(preprocess_threshold: int = 150) -> SRNoDetector:
    """Build a default SR number detector.

    Args:
        preprocess_threshold: Threshold value for binary thresholding.

    Returns:
        Configured SRNoDetector instance.
    """
    return SRNoDetector(preprocess_threshold=preprocess_threshold)
