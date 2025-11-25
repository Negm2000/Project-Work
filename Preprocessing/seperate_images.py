import shutil
import os
from pathlib import Path

# --- CONFIGURATION ---
RAW_IMAGES_DIR = Path("Datasets\\BLU_20251124") # Change this to your folder
OUTPUT_TOP = RAW_IMAGES_DIR / "TOP"
OUTPUT_BACK = RAW_IMAGES_DIR / "BACK"

def separate_images():
    OUTPUT_TOP.mkdir(exist_ok=True)
    OUTPUT_BACK.mkdir(exist_ok=True)
    
    # Find all png files
    files = list(RAW_IMAGES_DIR.glob("*.png"))
    print(f"Found {len(files)} images. Separating...")

    for f in files:
        # logic: Check if filename ends with TOP or BACK
        if "_TOP" in f.name.upper():
            shutil.move(str(f), str(OUTPUT_TOP / f.name))
        elif "_BACK" in f.name.upper():
            shutil.move(str(f), str(OUTPUT_BACK / f.name))

    print("Done! Check the 'TOP' and 'BACK' folders.")

if __name__ == "__main__":
    separate_images()