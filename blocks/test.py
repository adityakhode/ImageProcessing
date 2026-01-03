import subprocess
import glob

images = glob.glob("*.jpg")

for img in sorted(images):
    print(f"Running detect_grid.py on {img}")
    subprocess.run(["python", "detect_grid.py", img])
