import subprocess
import glob

# pdfs = glob.glob("*.pdf")
# print(pdfs)
# for pdf in sorted(pdfs):
#     print(f"Running detect_grid.py on {pdfs}")
#     subprocess.run(["python", "pdf_to_images.py", pdf])
pdfs = glob.glob("*.jpg")
print(len(pdfs))
