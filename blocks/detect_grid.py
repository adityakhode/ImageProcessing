#!/usr/bin/env python3
"""
Docstring for blocks.detect_grid
This Script is responsible for the detection of a grid for Individual voter blocks
from the pdf page and return the only grid Structure
"""

# Author: Aditya Khode
# Module: detect_grid.py
# Functions: get_grey_cv2_img, get_grid_from_grey_img, show_matplot_img, save_image
# Golbal Variable: None

import os                       # Use for path checks
import cv2                      # Image Operations
import sys                      # Quits the program safely when error occur
import numpy as np              # Use for Numpy image array operations and datatypes
import matplotlib.pyplot as plt # Use for plotting Image(Testing Purpose only)


#Class contains all the function thats helps to detect grid and test it.
class DETECT_GRID:

    def get_grey_cv2_img(self, img_path: str) -> np.ndarray:
        """
        Convert an image file to grayscale.
        
        This method reads an image from the specified path and converts it to a
        grayscale format for simplified image processing operations.
        
        Args:
            img_path (str): Absolute or relative path to the image file.
        
        Returns:
            np.ndarray: A 2D numpy array (uint8) representing the grayscale image.
        
        Raises:
            FileNotFoundError: If the image file does not exist.
            ValueError: If the image file cannot be read or is corrupted.
            
        Example:
            >>> detector = DETECT_GRID()
            >>> gray_img = detector.get_grey_cv2_img("path/to/image.jpg")
        """
        # Validate path exists
        if not os.path.exists(img_path):
            raise FileNotFoundError(f"Image file not found: {img_path}")
        
        # Read image
        img = cv2.imread(img_path)
        if img is None:
            raise ValueError(f"Failed to read image or unsupported format: {img_path}")
        
        # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        return gray

    def get_grid_from_grey_img(self, img: np.ndarray) -> np.ndarray:
        """
        Detect and extract grid lines from a grayscale image.
        
        Uses adaptive thresholding and morphological operations to identify
        horizontal and vertical grid lines, then combines them into a final grid.
        
        Args:
            img (np.ndarray): A 2D or 3D numpy array (grayscale or color image).
        
        Returns:
            np.ndarray: Binary image (uint8) with detected grid lines.
        
        Raises:
            ValueError: If img is None.
            TypeError: If img is not a numpy array.
        """
        # Validate input
        if img is None:
            raise ValueError("Input image cannot be None")

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


        # Apply adaptive threshold to convert to binary image
        bw = cv2.adaptiveThreshold(
            gray,
            255,
            cv2.ADAPTIVE_THRESH_MEAN_C,
            cv2.THRESH_BINARY_INV,
            25,
            10
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

    def show_matplot_img(self, img: np.ndarray) -> None:
        """
        Display an image using matplotlib.
        
        Args:
            img (np.ndarray): The image array to display.
        """
        plt.figure(figsize=(10, 14))
        plt.imshow(img, cmap="gray")
        plt.title("Detected Grid Lines")
        plt.axis("off")
        plt.show()

    def save_image(self, img_path: str, img_name: str, np_img: np.ndarray) -> None:
        """
        Save a numpy array as an image file.
        
        Args:
            img_path (str): Directory path where the image will be saved.
            img_name (str): Name of the image file (with extension).
            np_img (np.ndarray): The image array to save.
        """
        filename = os.path.basename(img_name)
        os.makedirs(img_path, exist_ok=True)
        output_path = os.path.join(img_path, filename)
        cv2.imwrite(output_path, np_img)

if __name__ == "__main__":
    img_path = "../data/BoothVoterList_A4_Ward_9_Booth_1_img001.jpg"

    try:
        detector = DETECT_GRID()
        
        # Get grey image
        grey_img = detector.get_grey_cv2_img(img_path)
        
        # Detect grid
        grid_img = detector.get_grid_from_grey_img(grey_img)
        
        # Show image
        detector.show_matplot_img(grid_img)
        
        # Save the image to desired file path
        detector.save_image("../data/output/", "detected_grid.jpg", grid_img)
        print("Grid detection completed successfully")
    
    except FileNotFoundError as e:
        print(f"Error: {e}")
        sys.exit(1)
    except ValueError as e:
        print(f"Error: {e}")
        sys.exit(1)