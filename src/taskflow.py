import os
import re
import csv
import glob
import time
import argparse
from pdf2image import convert_from_path
from page_processor import process_page
from concurrent.futures import ProcessPoolExecutor, as_completed
from config import (
    PDF_DIRS, OUTPUT_DIR, DPI, SKIP_FIRST_PAGE, SKIP_LAST_PAGE,
    NUM_WORKERS, MAX_WORKERS_CAP, ENABLE_CSV_EXPORT, CSV_FILENAME
)

class TASKFLOW:
    """
    Parallel PDF page processor with memory-optimized architecture.
    Processes pages concurrently using ProcessPoolExecutor.
    """
    
    def __init__(self):
        self.pdf_files = sorted(glob.glob(os.path.join(PDF_DIRS, "*.pdf")))
        self.num_workers = min(NUM_WORKERS, MAX_WORKERS_CAP)
        self.all_results = []
        self.metadata_defaults = {
            "epic_id": None,
            "name": None,
            "father_name": None,
            "husband_name": None,
            "block_code": None,
        }
        
        # Ensure output directory exists
        os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    def _prepare_page_data(self, pdf_name, pages):
        """
        Prepare page data for parallel processing.
        
        Returns:
            list of dicts with page metadata and image
        """
        page_data_list = []
        
        for idx, page in enumerate(pages):
            page_data_list.append({
                "pdf_name": os.path.basename(pdf_name),
                "page_num": idx + 1,
                "page_image": page
            })
        
        return page_data_list
    
    def _process_pdf(self, pdf_path):
        """
        Process single PDF file with parallel page processing.
        
        Args:
            pdf_path: path to PDF file
        """
        print(f"\n{'='*60}")
        print(f"Processing: {os.path.basename(pdf_path)}")
        print(f"{'='*60}")
        
        # Convert PDF to images
        pages = convert_from_path(pdf_path, dpi=DPI)
        
        # Skip first and last page if configured
        if SKIP_FIRST_PAGE:
            pages = pages[1:]
        if SKIP_LAST_PAGE:
            pages = pages[:-1]
        
        print(f"Total pages to process: {len(pages)}")
        
        # Prepare page data
        page_data_list = self._prepare_page_data(pdf_path, pages)
        del pages  # Free memory after preparation
        
        # Process pages in parallel
        pdf_results = []
        with ProcessPoolExecutor(max_workers=self.num_workers) as executor:
            # Submit all pages
            futures = {
                executor.submit(process_page, data): data["page_num"]
                for data in page_data_list
            }
            
            # Collect results as they complete
            completed = 0
            for future in as_completed(futures):
                page_num = futures[future]
                try:
                    page_results = future.result()
                    pdf_results.extend(page_results)
                    completed += 1
                    print(f"✓ Page {page_num} completed ({completed}/{len(page_data_list)})")
                except Exception as e:
                    print(f"✗ Page {page_num} failed: {e}")
        
        self.all_results.extend(pdf_results)
        print(f"Pages processed from PDF: {completed}/{len(page_data_list)}")

    def _validate_block_code(self, block_code: str) -> bool:
        """Validate block_code format like 42/178/796"""
        if block_code is None:
            return False
        return bool(re.match(r"^\d+\/\d+\/\d+$", block_code))

    def _process_photos(self, photos: list, metadata: dict):
        """Process a list of image paths with shared metadata (Teserack payload)."""
        page_data_list = []
        for i, p in enumerate(photos, start=1):
            # Read with OpenCV
            img = None
            try:
                import cv2
                img = cv2.imread(p)
            except Exception as e:
                print(f"✗ Failed to read image {p}: {e}")
                continue

            if img is None:
                print(f"✗ Image not found or unreadable: {p}")
                continue

            page_data_list.append({
                "pdf_name": metadata.get("epic_id") or os.path.basename(p),
                "page_num": i,
                "page_image": img
            })

        if not page_data_list:
            print("No valid photos to process")
            return

        # Process pages in parallel (re-using same executor logic)
        pdf_results = []
        with ProcessPoolExecutor(max_workers=self.num_workers) as executor:
            futures = {
                executor.submit(process_page, data): data["page_num"]
                for data in page_data_list
            }

            completed = 0
            for future in as_completed(futures):
                page_num = futures[future]
                try:
                    page_results = future.result()
                    # Attach metadata to each result row
                    for r in page_results:
                        r.update({
                            "epic_id": metadata.get("epic_id"),
                            "name": metadata.get("name"),
                            "father_name": metadata.get("father_name"),
                            "husband_name": metadata.get("husband_name"),
                            "block_code": metadata.get("block_code")
                        })
                    pdf_results.extend(page_results)
                    completed += 1
                    print(f"✓ Photo {page_num} completed ({completed}/{len(page_data_list)})")
                except Exception as e:
                    print(f"✗ Photo {page_num} failed: {e}")

        self.all_results.extend(pdf_results)
        print(f"Photos processed: {completed}/{len(page_data_list)}")
    
    def _save_results_to_csv(self):
        """Export all results to CSV file."""
        if not self.all_results or not ENABLE_CSV_EXPORT:
            return
        
        csv_path = os.path.join(OUTPUT_DIR, CSV_FILENAME)
        fieldnames = [
            "epic_id", "pdf_name", "page_num", "block_id", "age",
            "age_confidence", "gender", "gender_confidence",
            "name", "father_name", "husband_name", "block_code"
        ]
        
        try:
            with open(csv_path, 'w', newline='', encoding='utf-8') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(self.all_results)
            print(f"\n✓ Results saved to: {csv_path}")
            print(f"  Total records: {len(self.all_results)}")
        except Exception as e:
            print(f"✗ Failed to save CSV: {e}")
    
    def run(self):
        """Execute workflow: load PDFs sequentially, process pages in parallel."""
        if not self.pdf_files:
            print(f"No PDF files found in {PDF_DIRS}")
            return
        
        print(f"Found {len(self.pdf_files)} PDF file(s)")
        print(f"Using {self.num_workers} workers for parallel processing\n")
        
        # Process each PDF sequentially, but pages in parallel
        for pdf_path in self.pdf_files:
            self._process_pdf(pdf_path)
        
        # Save aggregated results
        self._save_results_to_csv()
        
        print(f"\n{'='*60}")
        print(f"Workflow complete. Total results: {len(self.all_results)}")
        print(f"{'='*60}\n")


def _parse_args():
    p = argparse.ArgumentParser(description="Run taskflow: process PDFs or a set of Teserack photos with metadata")
    p.add_argument("--photos", nargs="+", help="Paths to one or more photo files to process (images)")
    p.add_argument("--epic", help="Epic id supplied by Teserack")
    p.add_argument("--name", help="Subject name")
    p.add_argument("--father_name", help="Father's name")
    p.add_argument("--husband_name", help="Husband's name")
    p.add_argument("--block_code", help="Block code in format 42/178/796")
    p.add_argument("--skip-pdfs", action="store_true", help="Skip scanning the PDF_DIRS for PDFs")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    start_time = time.time()
    taskflow = TASKFLOW()

    if args.photos:
        metadata = {
            "epic_id": args.epic,
            "name": args.name,
            "father_name": args.father_name,
            "husband_name": args.husband_name,
            "block_code": args.block_code,
        }

        if metadata["block_code"] and not taskflow._validate_block_code(metadata["block_code"]):
            print(f"✗ Invalid block_code format: {metadata['block_code']} (expected N/N/N)")
        else:
            taskflow._process_photos(args.photos, metadata)

        # Save results if any
        taskflow._save_results_to_csv()

    else:
        if not args.skip_pdfs:
            taskflow.run()
        else:
            print("No photos provided and PDF processing skipped. Nothing to do.")

    end_time = time.time()
    elapsed = end_time - start_time
    print(f"Total execution time: {elapsed:.2f} seconds")