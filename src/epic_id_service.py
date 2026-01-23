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
from crop import crop_epic_id


@dataclass
class EpicIdDetectionResult:
    """Result of EPIC ID detection."""
    epic_id: Optional[str]
    confidence: float


class EpicIdDetector:
    """Detects and extracts EPIC ID from voter block images."""

    # EPIC ID format: 3 letters + 7 digits (e.g., ABC1234567)
    EPIC_ID_PATTERN = re.compile(r'[A-Z]{3}\d{7}', re.IGNORECASE)

    def __init__(self, preprocess_threshold: int = 150):
        """Initialize the EPIC ID detector.

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
        img = crop_epic_id(img)

        # Convert image to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # OPTIMIZATION: Skip bilateral filter (saves ~80-120ms per image)
        # Tesseract is robust enough for minor noise. Direct threshold is faster.
        # filtered = cv2.bilateralFilter(gray, 9, 75, 75)

        # Apply binary thresholding (convert to black and white)
        _, thresh = cv2.threshold(
            gray, self.preprocess_threshold, 255, cv2.THRESH_BINARY
        )

        # OPTIMIZATION: Skip morphological operations (saves ~20-30ms per image)
        # kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
        # thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=1)

        return thresh

    def _calculate_confidence(self, extracted_text: str, epic_id: str) -> float:
        """Calculate confidence score based on text quality.

        Args:
            extracted_text: Full extracted text from OCR.
            epic_id: Detected EPIC ID.

        Returns:
            Confidence score between 0.0 and 1.0.
        """
        # Base confidence
        confidence = 0.85
        
        # Increase confidence if EPIC ID is clearly visible in extracted text
        if epic_id.upper() in extracted_text.upper():
            confidence = 0.95
        
        return confidence

    def detect(self, img: np.ndarray) -> EpicIdDetectionResult:
        """Detect EPIC ID from image.

        Args:
            img: Input image (BGR format).

        Returns:
            EpicIdDetectionResult containing the detected EPIC ID and confidence.
        """
        if img is None:
            return EpicIdDetectionResult(epic_id=None, confidence=0.0)
        try:
            # Preprocess image
            preprocessed_img = self._preprocess_image(img)

            # Use Tesseract to extract text
            # The config `--psm 6` treats the image as a single uniform block of text
            extracted_text = pytesseract.image_to_string(
                preprocessed_img, config="--psm 6"
            )

            # Remove whitespace and special characters
            cleaned_text = re.sub(r'[\s\-]', '', extracted_text)

            # Use regex to find EPIC ID pattern (3 letters + 7 digits)
            match = self.EPIC_ID_PATTERN.search(cleaned_text)

            if not match:
                return EpicIdDetectionResult(epic_id=None, confidence=0.0)

            epic_id = match.group(0).upper()
            confidence = self._calculate_confidence(extracted_text, epic_id)

            return EpicIdDetectionResult(epic_id=epic_id, confidence=confidence)

        except Exception as e:
            print(f"Error detecting EPIC ID: {e}")
            return EpicIdDetectionResult(epic_id=None, confidence=0.0)


def build_default_detector(preprocess_threshold: int = 150) -> EpicIdDetector:
    """Build a default EPIC ID detector.

    Args:
        preprocess_threshold: Threshold value for binary thresholding.

    Returns:
        Configured EpicIdDetector instance.
    """
    return EpicIdDetector(preprocess_threshold=preprocess_threshold)