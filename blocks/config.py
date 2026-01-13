import os
from multiprocessing import cpu_count

# Directory paths
PDF_DIRS = os.path.join("..", "data", "pdfs")
OUTPUT_DIR = os.path.join("..", "data", "results")

# Processing parameters
DPI = 300
SKIP_FIRST_PAGE = True
SKIP_LAST_PAGE = True
MIN_AGE_THRESHOLD = 21

# Parallel processing
NUM_WORKERS = max(1, cpu_count() - 1)  # Leave one core free
MAX_WORKERS_CAP = 8  # Prevent memory saturation

# Output
ENABLE_CSV_EXPORT = True
CSV_FILENAME = "detection_results.csv"
