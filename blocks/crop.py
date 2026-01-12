import cv2

def crop_age(img):
    h, w = img.shape[:2]

    x1 = int(0.088 * w)
    x2 = int(0.16  * w)
    y1 = int(0.83  * h)
    y2 = int(0.96  * h)

    if x2 <= x1 or y2 <= y1:
        raise ValueError("Invalid crop region")

    return img[y1:y2, x1:x2]
