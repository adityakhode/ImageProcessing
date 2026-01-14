# ImageProcessing — Crop Utilities

## Purpose

This repository contains utilities for extracting regions of interest from
scanned documents or images (age/gender fields and tight crops around
foreground content). It helps preprocess images for downstream OCR or
classification models.

## Features

- Deterministic crops for age and gender regions.
- Tight crop around dark foreground pixels (suitable for scanned digits/text).
- Robust validation and safe image slicing behavior.

## Tech Stack

- Python 3.10+
- OpenCV (`cv2`)
- NumPy

## Setup

1. Create and activate a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate
```

2. Install dependencies:

```bash
pip install -r requirments.txt
```

3. Run quick interactive test (example):

```python
from blocks import crop
import cv2

img = cv2.imread('data/pdfs/example.png')
age_region = crop.crop_age(img)
gender_region = crop.crop_gender(img)
cropped = crop.cut2cut_crop(img)
```

## Configuration

There are no runtime secrets required for the cropping utilities. If you
extend the project (for model inference, cloud storage, etc.) list env
vars and secret management here.

## Folder Structure

- `blocks/` : image-processing modules (including `crop.py`).
- `data/` : sample inputs and outputs.
- `digits/` : templates and digit-related resources.
- `notebook_archive/` : exploratory notebooks.

## Known Limitations

- The deterministic crop ratios (`crop_age`, `crop_gender`) are tuned for a
  specific document layout and may fail for other templates.
- `cut2cut_crop` assumes a light background and darker foreground; it will
  not work well with noisy or multi-colored backgrounds.

## Next Steps to Make This Project More Contributable

- Add `CONTRIBUTING.md` with contribution guidelines and `CODE_OF_CONDUCT`.
- Add unit tests for `blocks/crop.py` and CI (GitHub Actions) to run them.
- Add a `LICENSE` file (e.g., MIT) to enable open-source contributions.
- Add type-checking via `mypy` and a linter config (e.g., `flake8`/`ruff`).

---

See `src/crop.py` for detailed documentation of the functions.

