#!/usr/bin/env python3
"""
DEBUG CROPS
===========
Extracts random samples from the dataset and saves visualizations of:
1. Full image with ROI drawn
2. Extracted BGR crop
3. Normalized crop (as seen by the model)
"""

import os, sys, json, random, cv2
import numpy as np
from pathlib import Path
from dataclasses import dataclass
import time

SCRIPT_DIR = Path(__file__).resolve().parent
PACKAGE_DIR = SCRIPT_DIR / "BEKO_PCL_Presence_System_Complete"
DATA_DIR = SCRIPT_DIR / "Data" / "aligned_top"
ROI_CONFIG = PACKAGE_DIR / "config" / "roi_config.json"
DATASET_CSV = SCRIPT_DIR / "dataset.csv"
OUTPUT_DIR = SCRIPT_DIR / "outputs" / "debug_visuals"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

@dataclass
class RelativeROI:
    name: str
    x_min_rel: float
    y_min_rel: float
    x_max_rel: float
    y_max_rel: float
    
    def to_pixel_box(self, width: int, height: int, margin: int = 0):
        x_min = int(self.x_min_rel * width)
        y_min = int(self.y_min_rel * height)
        x_max = int(self.x_max_rel * width)
        y_max = int(self.y_max_rel * height)
        x_min = max(0, x_min - margin)
        y_min = max(0, y_min - margin)
        x_max = min(width, x_max + margin)
        y_max = min(height, y_max + margin)
        return x_min, y_min, x_max, y_max

def load_roi_config(roi_config_path):
    with open(roi_config_path, 'r') as f:
        data = json.load(f)
    rois = []
    for entry in data:
        rois.append(RelativeROI(
            name=entry["name"],
            x_min_rel=float(entry["x_min_rel"]),
            y_min_rel=float(entry["y_min_rel"]),
            x_max_rel=float(entry["x_max_rel"]),
            y_max_rel=float(entry["y_max_rel"])
        ))
    return rois

def normalize_roi(img: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    equalized = clahe.apply(gray)
    normalized = equalized.astype(np.float32) / 255.0
    return normalized

def extract_connectors(aligned_image, rois, margin=10):
    height, width = aligned_image.shape[:2]
    connectors = []
    
    for roi in rois:
        x_min, y_min, x_max, y_max = roi.to_pixel_box(width, height, margin=margin)
        crop_bgr = aligned_image[y_min:y_max, x_min:x_max].copy()
        
        # Normalizza come in crop_connectors.py (grayscale + CLAHE)
        normalized = normalize_roi(crop_bgr)
        
        # Converti a uint8 per salvare/visualizzare
        crop_normalized = (normalized * 255).astype(np.uint8)
        
        connectors.append({
            'name': roi.name,
            'crop': crop_normalized,  # Grayscale normalizzato
            'crop_bgr': crop_bgr,  # Mantieni BGR per visualizzazione originale
            'bbox': (x_min, y_min, x_max, y_max)
        })
    
    return connectors

def main():
    import pandas as pd
    
    if not DATASET_CSV.exists():
        print(f"Dataset CSV not found: {DATASET_CSV}")
        return

    print(f"Loading dataset from {DATASET_CSV}...")
    try:
        df = pd.read_csv(DATASET_CSV)
        # Filter for relevant labels only
        valid_labels = ["OK", "KO"]
        df = df[df["label"].isin(valid_labels)]
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return
        
    rois = load_roi_config(ROI_CONFIG)
    
    # Process unique connectors
    connectors = df["connector_name"].unique()
    
    for cid in connectors:
        print(f"Visualizing {cid}...")
        
        # Get random samples for this connector
        subset = df[df["connector_name"] == cid]
        if subset.empty:
            continue
            
        sample_rows = subset.sample(min(3, len(subset)))
        
        # Find ROI
        current_roi = next((r for r in rois if r.name == cid), None)
        if not current_roi:
            print(f"  Missing ROI config for {cid}")
            continue
            
        for idx, row in sample_rows.iterrows():
            # Get filename from path
            full_drive_path = row["image_path"]
            fname = Path(full_drive_path).name 
            full_path = DATA_DIR / fname
            label = row["label"]
            
            if not full_path.exists():
                print(f"  Image not found: {full_path}")
                continue
                
            image_bgr = cv2.imread(str(full_path))
            if image_bgr is None:
                continue
            
            # Extract
            extracted_list = extract_connectors(image_bgr, [current_roi], margin=10)
            if not extracted_list:
                continue
                
            data = extracted_list[0]
            crop_norm = data['crop']
            crop_bgr = data['crop_bgr']
            bbox = data['bbox']
            
            # Save files
            base_name = f"{cid}_{label}_{idx}"
            
            # 1. Normalized crop (what model sees)
            path_norm = OUTPUT_DIR / f"{base_name}_norm.png"
            cv2.imwrite(str(path_norm), crop_norm)
            
            # 2. BGR crop (human readable)
            path_bgr = OUTPUT_DIR / f"{base_name}_bgr.png"
            cv2.imwrite(str(path_bgr), crop_bgr)
            
            # 3. Full image with box
            img_bbox = image_bgr.copy()
            cv2.rectangle(img_bbox, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 0, 255), 3)
            path_bbox = OUTPUT_DIR / f"{base_name}_full.png"
            cv2.imwrite(str(path_bbox), img_bbox)
            
            print(f"  Saved debug visuals for {base_name}")

    print(f"\n✅ All visuals saved to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
