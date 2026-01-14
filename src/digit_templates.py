import os
import cv2
from typing import Dict


def load_templates(template_dir: str = "../digits/templates/") -> Dict[int, "cv2.UMat"]:
    """Load digit templates (0-9) from `template_dir`.

    Returns a dict mapping int digit -> grayscale numpy array.
    Raises RuntimeError if any template is missing.
    """
    templates = {}
    for i in range(10):
        path = os.path.join(template_dir, f"{i}.png")
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise RuntimeError(f"Template missing: {path}")
        templates[i] = img
    return templates
