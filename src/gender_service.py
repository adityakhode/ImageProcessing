from dataclasses import dataclass
from typing import Optional
import numpy as np

from gender_templates import load_gender_templates
from gender_matcher import SSIMGenderMatcher


@dataclass
class GenderDetectionResult:
    gender: Optional[str]
    confidence: float


class GenderDetector:
    def __init__(self, matcher: SSIMGenderMatcher):
        self.matcher = matcher

    def detect(self, img: np.ndarray) -> GenderDetectionResult:
        gender, score = self.matcher.match(img)
        return GenderDetectionResult(gender=gender, confidence=score)


def build_default_gender_detector(template_dir: str = "../digits/templates/") -> GenderDetector:
    templates = load_gender_templates(template_dir)
    matcher = SSIMGenderMatcher(templates)
    return GenderDetector(matcher)
