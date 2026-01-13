import cv2
import os
from split_numbers import split_img, resize
from skimage.metrics import structural_similarity as ssim

TEMPLATE_DIR = "../digits/templates/"
STANDARD_SIZE = 64   # single int is easier for centering

def load_templates():
    templates = {}
    for i in range(10):
        path = os.path.join(TEMPLATE_DIR, f"{i}.png")
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE) 
        if img is None:
            raise RuntimeError(f"Template missing: {path}")
        templates[i] = img
    return templates

TEMPLATES = load_templates()

def detect_age(img):
    age = ""

    images = split_img(img)
    confidence_score_avg = 0
    for image in (images):
        digit, confidence = detect_digit(resize(image))
        confidence_score_avg += confidence
        age = age + str(digit)
    return int(age), confidence_score_avg/2

def detect_digit(img):
    best_score = -1
    best_digit = None

    for digit, template in TEMPLATES.items():
        score = ssim(img, template)

        if score > best_score:
            best_score = score
            best_digit = digit

    return best_digit, best_score


if __name__ == "__main__":
    path = "../data/output/number8.jpg"
    img = cv2.imread(path)
    age = detect_age(img)
    print(age)
    # images = detect_age(img)
    
    # count = 0
    # for i, image in enumerate(images):
    #     digit, confidence = detect_digit(image)
    #     cv2.imshow(f"{digit}", image)
    #     cv2.waitKey(3000)
    #     cv2.imwrite(f"../data/output/{digit}.png",image)
    #     count+=1
    #     cv2.destroyAllWindows()
    #     print(f"[{i}] Detected Digit:", digit)
    #     print(f"[{i}] Confidence:", round(confidence, 4))

