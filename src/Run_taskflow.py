import os
import glob
import time
from detect_grid import process_img
from multiprocessing import Pool, cpu_count

if __name__ == "__main__":
    start_time = time.time()
    images = sorted(glob.glob("../data/*.jpg"))

    os.makedirs("../data/output", exist_ok=True)

    workers = max(1, cpu_count() - 1)
    print(f"Using {workers} workers")

    with Pool(workers) as pool:
        results = pool.map(process_img, images)

    end_time = time.time()
    count = 0
    for r in results:
        count +=1
    print(f"Processed {count} in {end_time - start_time} sec")