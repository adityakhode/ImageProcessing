import cv2
import numpy as np
import matplotlib
matplotlib.use("TkAgg")
import argparse
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser(description="Print a passed argument")
parser.add_argument("text", help="String to print")

args = parser.parse_args()

path = args.text
# Read image
img = cv2.imread(path)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

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

    if width > w * 0.45 and area > 800:
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
vertical = cv2.dilate(vertical, np.ones((15, 3), np.uint8), iterations=1)

num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(vertical, 8)
clean_vertical = np.zeros_like(vertical)

for i in range(1, num_labels):
    height = stats[i, cv2.CC_STAT_HEIGHT]
    area = stats[i, cv2.CC_STAT_AREA]

    if height > h * 0.45 and area > 800:
        clean_vertical[labels == i] = 255

# -------------------------
# FINAL GRID
# -------------------------
grid = cv2.bitwise_or(clean_horizontal, clean_vertical)

# -------------------------
# DISPLAY USING MATPLOTLIB
# -------------------------
plt.figure(figsize=(10, 14))
plt.imshow(grid, cmap="gray")
plt.title("Detected Grid Lines")
plt.axis("off")

# This blocks execution until you close the window
plt.show()
