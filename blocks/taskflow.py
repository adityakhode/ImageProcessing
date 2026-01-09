from pdf2image import convert_from_path
import os
import cv2
import numpy as np
import time
import glob
import os
import gc
from detect_grid import get_grid_from_grey_img
from extract_grid_coordinates import get_coordinates
from detect_age import detect_number
pdf_dir = "../data/pdfs"

pdf_files = sorted(glob.glob(os.path.join(pdf_dir, "*.pdf")))

del pdf_dir
gc.collect()

page = 1
for pdf_name in pdf_files:
    # Convert all pages
    pages = convert_from_path(pdf_name, dpi=300)
    
    record_count = 0
    page_count = 0
    start_time = time.time()
    # Remove first and last page
    for page in pages[1:-1]:  # skip first & last page
        page_count+=1
        img_orignal = np.array(page)
        img = cv2.cvtColor(img_orignal, cv2.COLOR_RGB2BGR)
        
        grid_img = get_grid_from_grey_img(img)
        rectangles = get_coordinates(grid_img)
        for rect_name, rect in rectangles.items():
            # Extract coordinates
            x1, y1 = rect["top_left"]
            x2, y2 = rect["bottom_right"]

            # Ensure integers
            x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])

            # Crop
            crop = img_orignal[y1:y2, x1:x2]
            record_count +=1
            # -------------------------
            # SHOW cropped IMAGE
            # -------------------------
            STANDARD_W = 1200
            STANDARD_H = 600

            crop = cv2.resize(crop, (STANDARD_W, STANDARD_H))

            age, crop = detect_number(crop)
            print("age", age)
            cv2.putText(crop, age, (12, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2, cv2.LINE_AA)
            cv2.imshow("Grid Points", crop)
            cv2.waitKey(2000)
            cv2.destroyAllWindows()
            
    end_time = time.time()
    print(f"1 pdf having {page_count} pages and {record_count} candidates cropped in {end_time - start_time}")