import os
import cv2
from typing import Dict


def load_gender_templates(template_dir: str = "../digits/templates/") -> Dict[str, "cv2.UMat"]:
    """Load gender templates 'M' and 'F' from `template_dir`.

    Returns a dict mapping 'M'/'F' -> grayscale numpy array.
    Raises RuntimeError if any template is missing.
    """
    templates = {}
    for gender in ("M", "F"):
        path = os.path.join(template_dir, f"{gender}.png")
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise RuntimeError(f"Template missing: {path}")
        templates[gender] = img
    return templates
