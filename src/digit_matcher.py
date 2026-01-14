from abc import ABC, abstractmethod
from typing import Dict, List, Tuple
import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim


class DigitMatcher(ABC):
    @abstractmethod
    def match(self, img: np.ndarray) -> Tuple[int, float]:
        """Return (digit, score) for a single preprocessed image."""


class SSIMMatcher(DigitMatcher):
    def __init__(self, templates: Dict[int, np.ndarray], standard_size: int = 64):
        self.standard_size = standard_size
        # Pre-resize templates to standard size for faster matching
        self.templates = {
            d: cv2.resize(tpl, (standard_size, standard_size))
            for d, tpl in templates.items()
        }

    def match(self, img: np.ndarray) -> Tuple[int, float]:
        # Ensure grayscale and correct size
        if img.ndim == 3:
            img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            img_gray = img

        img_resized = cv2.resize(img_gray, (self.standard_size, self.standard_size))

        best_score = -1.0
        best_digit = None

        for digit, tpl in self.templates.items():
            try:
                score = ssim(img_resized, tpl)
            except ValueError:
                score = -1.0

            if score > best_score:
                best_score = score
                best_digit = digit

        return best_digit, float(best_score)
