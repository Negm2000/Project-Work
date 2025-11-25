import cv2
import json
import os
from pathlib import Path

# --- CONFIGURATION ---
ALIGNED_DIR = Path("Data/aligned_top")
CONFIG_PATH = Path("Preprocessing/roi_config.json")
OUTPUT_FILE = "debug_visualization.jpg"

def draw_rois():
    # 1. Load the ROI config
    with open(CONFIG_PATH, 'r') as f:
        rois = json.load(f)

    # 2. Get the first image from your aligned folder
    images = list(ALIGNED_DIR.glob("*.png"))
    if not images:
        print("Error: No images found in Data/aligned_top/")
        return

    # Use the first image found (or your specific reference if you know it)
    target_img_path = images[0]
    print(f"Debugging using image: {target_img_path}")
    
    img = cv2.imread(str(target_img_path))
    height, width = img.shape[:2]

    # 3. Draw the boxes
    for roi in rois:
        # Convert relative (0-1) to pixels
        x1 = int(roi['x_min_rel'] * width)
        y1 = int(roi['y_min_rel'] * height)
        x2 = int(roi['x_max_rel'] * width)
        y2 = int(roi['y_max_rel'] * height)

        # Draw rectangle (Blue)
        cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 3)
        
        # Draw label
        cv2.putText(img, roi['name'], (x1, y1-10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

    # 4. Save the result
    cv2.imwrite(OUTPUT_FILE, img)
    print(f"Saved visualization to {OUTPUT_FILE}. Please open this image!")

if __name__ == "__main__":
    draw_rois()