from crop import crop_house_no
import cv2
# Read image
img = cv2.imread(r"../data/output/1.png")

crop = crop_house_no(img)

cv2.imwrite(r"../data/output/1_name_crop.png", crop)