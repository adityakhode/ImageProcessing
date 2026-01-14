from abc import ABC, abstractmethod
from typing import Dict, Tuple
import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim


class GenderMatcher(ABC):
    @abstractmethod
    def match(self, img: np.ndarray) -> Tuple[str, float]:
        """Return (gender, score) where gender is 'M' or 'F'."""


class SSIMGenderMatcher(GenderMatcher):
    def __init__(self, templates: Dict[str, np.ndarray], standard_size: int = 64):
        self.standard_size = standard_size
        self.templates = {
            g: cv2.resize(tpl, (standard_size, standard_size))
            for g, tpl in templates.items()
        }

    def match(self, img: np.ndarray) -> Tuple[str, float]:
        if img.ndim == 3:
            img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            img_gray = img

        img_resized = cv2.resize(img_gray, (self.standard_size, self.standard_size))

        best_score = -1.0
        best_gender = None
        for gender, tpl in self.templates.items():
            try:
                score = ssim(img_resized, tpl)
            except ValueError:
                score = -1.0

            if score > best_score:
                best_score = score
                best_gender = gender

        return best_gender, float(best_score)
