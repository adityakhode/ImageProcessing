import os
import glob
import time
import csv
from concurrent.futures import ProcessPoolExecutor, as_completed
from pdf2image import convert_from_path
from page_processor import process_page
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
        self.num_workers = min(NUM_WORKERS + 1, MAX_WORKERS_CAP)
        self.all_results = []
        
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
    
    def _save_results_to_csv(self):
        """Export all results to CSV file."""
        if not self.all_results or not ENABLE_CSV_EXPORT:
            return
        
        csv_path = os.path.join(OUTPUT_DIR, CSV_FILENAME)
        fieldnames = ["pdf_name", "page_num", "block_id", "age", 
                     "age_confidence", "gender", "gender_confidence"]
        
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


# Entry point
if __name__ == "__main__":
    start_time = time.time()
    taskflow = TASKFLOW()
    taskflow.run()
    end_time = time.time()
    elapsed = end_time - start_time
    print(f"Total execution time: {elapsed:.2f} seconds")