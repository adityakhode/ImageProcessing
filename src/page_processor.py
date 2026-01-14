import numpy as np
from detect_age import detect_age
from detect_grid import DETECT_GRID
from config import MIN_AGE_THRESHOLD
from detect_gender import detect_gender
from extract_grid_coordinates import get_coordinates


def process_page(page_data):
    """
    Process single PDF page: grid detection → block extraction → age/gender detection.
    Returns list of detection results for all blocks in page.
    
    Args:
        page_data: dict with keys {pdf_name, page_num, page_image}
    
    Returns:
        list of dicts with detection results or empty list on error
    """
    try:
        pdf_name = page_data["pdf_name"]
        page_num = page_data["page_num"]
        page_image = page_data["page_image"]
        
        # Convert to numpy array if needed
        original_page = np.array(page_image)
        results = []
        
        # Initialize grid detector (fresh instance per worker)
        grid_detector = DETECT_GRID()
        
        # Step 1: Detect grid in page
        grid_image = grid_detector.get_grid_from_grey_img(original_page)
        del grid_detector  # Free memory immediately
        
        # Step 2: Extract grid coordinates
        rectangles = get_coordinates(grid_image)
        del grid_image  # Free memory after extraction
        
        # Step 3: Process each block in grid
        for block_id, rect in rectangles.items():
            try:
                # Extract coordinates
                x1, y1 = rect["top_left"]
                x2, y2 = rect["bottom_right"]
                x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
                
                # Crop voter block
                voter_block = original_page[y1:y2, x1:x2]
                
                # Detect age
                age, age_conf = detect_age(voter_block)
                if age is None or (isinstance(age, int) and age < MIN_AGE_THRESHOLD):
                    age_val = "NULL"
                else:
                    age_val = age
                
                # Detect gender
                gender, gender_conf = detect_gender(voter_block)
                
                # Collect result
                results.append({
                    "pdf_name": pdf_name,
                    "page_num": page_num,
                    "block_id": block_id,
                    "age": age_val,
                    "age_confidence": round(age_conf, 4),
                    "gender": gender,
                    "gender_confidence": round(gender_conf, 4)
                })
            except Exception as block_error:
                print(f"Error processing block {block_id} on page {page_num}: {block_error}")
                continue
        
        return results
    
    except Exception as page_error:
        print(f"Error processing page {page_data['page_num']} from {page_data['pdf_name']}: {page_error}")
        return []
