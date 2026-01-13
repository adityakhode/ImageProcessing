import os
import cv2
import numpy as np
from skimage import color, filters
from crop import crop_gender, cut2cut_crop
from skimage.metrics import structural_similarity as ssim

STANDARD_SIZE = 64

def resize(img: np.ndarray) -> np.ndarray:
    return cv2.resize(img, (STANDARD_SIZE, STANDARD_SIZE))

TEMPLATE_DIR = "../digits/templates/"

def load_templates():
    templates = {}
    genders = ['M', 'F']
    for gender in genders:
        path = os.path.join(TEMPLATE_DIR, f"{gender}.png")
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE) 
        if img is None:
            raise RuntimeError(f"Template missing: {path}")
        templates[gender] = img
    return templates


TEMPLATES = load_templates()

def guess_gender(img):
    best_score = -1
    best_gender = None

    for gender, template in TEMPLATES.items():
        score = ssim(img, template)

        if score > best_score:
            best_score = score
            best_gender = gender

    return best_gender, best_score

def detect_gender(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    cropped_gender = crop_gender(img)
    cut2cutCropped_img = cut2cut_crop(cropped_gender)
    img_p = resize(cut2cutCropped_img)
    
    _, binary = cv2.threshold(img_p, 45, 255, cv2.THRESH_BINARY)
    gender, confidence_score = guess_gender(binary)
    return gender, confidence_score
    

if __name__ == "__main__":

    path = r"../data/output/number843.jpg"
    img = cv2.imread(path, cv2.IMREAD_REDUCED_GRAYSCALE_2)

    gender, gender_confidence_score = detect_gender(img)
    print("Gender ", gender, " Gender confidence_score ", gender_confidence_score)


    # cropped_gender = crop_gender(img)
    # cut2cutCropped_img = cut2cut_crop(cropped_gender)
    # img = resize(cut2cutCropped_img)
    
    
    # # Create kernel (structuring element)
    # kernel = np.ones((5, 5), np.uint8)

    # # Apply erosion
    # eroded = cv2.erode(img, kernel, iterations=1)
    # _, binary = cv2.threshold(eroded, 45, 255, cv2.THRESH_BINARY)
    
    # cv2.imwrite("../data/output/crop.png", binary)