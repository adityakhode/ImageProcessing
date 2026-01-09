import os
import cv2
import sys
import numpy as np
import matplotlib.pyplot as plt

def get_grey_cv2_img(img_path: str):
    img = cv2.imread(img_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return gray

def get_grid_from_grey_img(img):
    # ------------------ HARD NORMALIZATION ------------------
    if img is None:
        raise ValueError("process_img received None")

    if not isinstance(img, np.ndarray):
        raise TypeError(f"Expected numpy array, got {type(img)}")

    # Convert to grayscale if needed
    if img.ndim == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img

    # Force uint8 (THIS IS THE KEY FIX)
    if gray.dtype != np.uint8:
        gray = gray.astype(np.uint8)


    bw = cv2.adaptiveThreshold(
        gray,
        255,
        cv2.ADAPTIVE_THRESH_MEAN_C,
        cv2.THRESH_BINARY_INV,
        25,
        10
    )
    # Adaptive threshold
    bw = cv2.adaptiveThreshold(
        gray, 255,
        cv2.ADAPTIVE_THRESH_MEAN_C,
        cv2.THRESH_BINARY_INV,
        25, 10
    )

    h, w = bw.shape
    # -------------------------
    # HORIZONTAL LINE DETECTION
    # -------------------------
    h_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (w // 12, 1))
    horizontal = cv2.morphologyEx(bw, cv2.MORPH_OPEN, h_kernel)

    # Repair broken lines
    horizontal = cv2.dilate(horizontal, np.ones((3, 15), np.uint8), iterations=1)

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(horizontal, 8)
    clean_horizontal = np.zeros_like(horizontal)

    for i in range(1, num_labels):
        width = stats[i, cv2.CC_STAT_WIDTH]
        area = stats[i, cv2.CC_STAT_AREA]

        if width > w * 0.30 and area > 800:
            clean_horizontal[labels == i] = 255

    # Remove bottom-most horizontal line (footer)
    ys = np.where(clean_horizontal.sum(axis=1) > 0)[0]
    if len(ys) > 0 and ys[-1] > h * 0.92:
        y = ys[-1]
        clean_horizontal[max(0, y-3):min(h, y+3), :] = 0

    # -------------------------
    # VERTICAL LINE DETECTION
    # -------------------------
    v_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, h // 12))
    vertical = cv2.morphologyEx(bw, cv2.MORPH_OPEN, v_kernel)

    # Repair broken lines
    clean_vertical = cv2.dilate(vertical, np.ones((15, 3), np.uint8), iterations=1)

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(vertical, 8)

    for i in range(1, num_labels):
        height = stats[i, cv2.CC_STAT_HEIGHT]
        area = stats[i, cv2.CC_STAT_AREA]

        if height > h * 0.25 and area > 800:
            clean_vertical[labels == i] = 255


    # -------------------------
    # FINAL GRID
    # -------------------------
    grid = cv2.bitwise_or(clean_horizontal, clean_vertical)

    return grid

def show_matplot_img(img):
        # # -------------------------
    # DISPLAY USING MATPLOTLIB
    # -------------------------
    plt.figure(figsize=(10, 14))
    plt.imshow(img, cmap="gray")
    plt.title("Detected Grid Lines")
    plt.axis("off")
    plt.show()

def save_image(img_path: str, img_name: str, np_img):
    filename = os.path.basename(img_name)
    os.makedirs(img_path, exist_ok=True)
    output_path = os.path.join(img_path, filename)
    cv2.imwrite(output_path, np_img)

if __name__ == "__main__":
    img_path = "../data/BoothVoterList_A4_Ward_9_Booth_1_img001.jpg"

    if not os.path.exists(img_path):
        print("No Img found")
        sys.exit(1)

    #Get grey image
    grey_img = get_grey_cv2_img(img_path)

    #detect Grid
    grid_img = get_grid_from_grey_img(grey_img)

    #Show Image
    show_matplot_img(grid_img)

    #Save the image to desired file path
    save_image("../data/output/", "detected_grid.jpg", grid_img)