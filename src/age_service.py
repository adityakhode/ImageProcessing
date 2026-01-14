from dataclasses import dataclass
from typing import Optional, List, Tuple
import numpy as np

from digit_matcher import DigitMatcher, SSIMMatcher
from digit_templates import load_templates
from split_numbers import split_img


@dataclass
class AgeDetectionResult:
    age: Optional[int]
    confidence: float


class AgeDetector:
    """Orchestrates splitting and matching to return an age and confidence."""

    def __init__(self, matcher: DigitMatcher):
        self.matcher = matcher

    def detect(self, img: np.ndarray) -> AgeDetectionResult:
        parts = split_img(img)

        if not parts:
            return AgeDetectionResult(age=0, confidence=0.0)

        digits: List[str] = []
        scores: List[float] = []

        for part in parts:
            digit, score = self.matcher.match(part)
            digits.append(str(digit))
            scores.append(score)

        try:
            age_value = int("".join(digits))
        except Exception:
            age_value = None

        avg_conf = float(sum(scores) / max(1, len(scores)))
        return AgeDetectionResult(age=age_value, confidence=avg_conf)


def build_default_detector(template_dir: str = "../digits/templates/") -> AgeDetector:
    templates = load_templates(template_dir)
    matcher = SSIMMatcher(templates)
    return AgeDetector(matcher)
