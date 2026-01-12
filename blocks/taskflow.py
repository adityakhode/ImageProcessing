import os
import gc
import cv2
import time
import glob
import numpy as np
from detect_age import detect_age
from pdf2image import convert_from_path
from detect_grid import get_grid_from_grey_img
from extract_grid_coordinates import get_coordinates

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
    negative_marking = 0

    start_time = time.time()
    # Remove first and last page
    for page in pages[1:-1]:  # skip first & last page
        page_count+=1
        img_orignal = np.array(page)
        pages.pop(0)
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

            age, confidence_score = detect_age(crop)
            print("record_count ", record_count, " age ", age, " confidence_score ", confidence_score)

            cv2.putText(crop,str(age),(20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), cv2.LINE_AA)
            # cv2.imshow("Text", crop)
            # cv2.waitKey(0)
            # cv2.destroyAllWindows()
            if confidence_score < 0.7:
                cv2.imwrite(f"../data/output/number{record_count}.jpg", crop)



            
    end_time = time.time()
    print(f"1 pdf having {page_count} pages and {record_count} candidates cropped in {end_time - start_time}")