import os
import sys
import cv2
import numpy as np

def get_cv2_img(grid_path: str):
    grid = cv2.imread(grid_path)

    if grid is None:
        raise ValueError("Could not load image")
    return grid

def cluster_positions(positions, gap=5):
    clusters = []
    current = [positions[0]]
    for p in positions[1:]:
        if p - current[-1] <= gap:
            current.append(p)
        else:
            clusters.append(int(np.mean(current)))
            current = [p]

    clusters.append(int(np.mean(current)))
    return clusters

def get_coordinates(cv2_img):
    # _, grid = cv2.threshold(cv2_img, 127, 255, cv2.THRESH_BINARY)
    # h, w = grid.shape
    grid = cv2_img

    # ---- Horizontal lines ----
    h_proj = np.sum(grid, axis=1)
    h_thresh = 0.5 * np.max(h_proj)
    ys = np.where(h_proj > h_thresh)[0]
    row_lines = cluster_positions(ys)

    # ---- Vertical lines ----
    v_proj = np.sum(grid, axis=0)
    v_thresh = 0.5 * np.max(v_proj)
    xs = np.where(v_proj > v_thresh)[0]
    col_lines = cluster_positions(xs)

    rectangles = {}
    rect_id = 1

    for r in range(len(row_lines) - 1):
        for c in range(len(col_lines) - 1):
            x1, x2 = col_lines[c], col_lines[c + 1]
            y1, y2 = row_lines[r], row_lines[r + 1]

            rectangles[f"rectangle_{rect_id}"] = {
                "top_left": [x1, y1],
                "top_right": [x2, y1],
                "bottom_right": [x2, y2],
                "bottom_left": [x1, y2]
            }
            rect_id += 1
    return rectangles

if __name__ == "__main__":
    grid_path = "../data/output/detected_grid.jpg"

    if not os.path.exists(grid_path):
        print("No img found")
        sys.exit(1)
    grid_img = get_cv2_img(grid_path)
    rectangles = get_coordinates(grid_img)
    print(rectangles)
