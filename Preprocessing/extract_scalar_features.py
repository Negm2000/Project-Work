from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from typing import Iterable, List, Optional

import cv2
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm


SUPPORTED_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
LABEL_PATTERN = re.compile(r"_(OK|KO)(?:[^a-zA-Z0-9]|$)", re.IGNORECASE)


@dataclass
class BandConfig:
    rel_ymin: float
    rel_ymax: float

    def validate(self) -> None:
        if not (0.0 <= self.rel_ymin < self.rel_ymax <= 1.0):
            raise ValueError(
                "Band relative coordinates must satisfy 0 <= ymin < ymax <= 1."
            )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract simple scalar features from connector ROIs."
    )
    parser.add_argument(
        "--input_dir",
        type=Path,
        required=True,
        help="Root directory containing connector folders.",
    )
    parser.add_argument(
        "--output_csv",
        type=Path,
        required=True,
        help="Path to the CSV file to write features into.",
    )
    parser.add_argument(
        "--band_rel_ymin",
        type=float,
        default=None,
        help="Lower relative Y coordinate (0-1) for the horizontal band.",
    )
    parser.add_argument(
        "--band_rel_ymax",
        type=float,
        default=None,
        help="Upper relative Y coordinate (0-1) for the horizontal band.",
    )
    return parser.parse_args()


def find_images(root: Path) -> List[Path]:
    if not root.exists():
        raise FileNotFoundError(f"Input directory '{root}' does not exist.")

    files: List[Path] = [
        path
        for path in sorted(root.rglob("*"))
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS
    ]
    if not files:
        raise FileNotFoundError(f"No supported image files found under '{root}'.")
    return files


def load_normalized_gray(image_path: Path) -> np.ndarray:
    img = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"Failed to load image '{image_path}'.")
    if img.dtype != np.float32:
        img = img.astype(np.float32) / 255.0
    return np.clip(img, 0.0, 1.0)


def detect_label(filename: str) -> Optional[str]:
    match = LABEL_PATTERN.search(filename)
    if match:
        return match.group(1).upper()
    return None


def compute_band_indices(image: np.ndarray, band: BandConfig) -> slice:
    height = image.shape[0]
    y_min = int(np.floor(band.rel_ymin * height))
    y_max = int(np.ceil(band.rel_ymax * height))
    y_min = max(0, min(height, y_min))
    y_max = max(y_min + 1, min(height, y_max))
    return slice(y_min, y_max)


def build_records(
    image_paths: Iterable[Path],
    band: Optional[BandConfig],
) -> List[dict]:
    records: List[dict] = []
    for img_path in tqdm(image_paths, desc="Processing ROIs"):
        image = load_normalized_gray(img_path)
        gray_mean = float(np.mean(image))
        gray_std = float(np.std(image))

        record = {
            "sample_id": img_path.stem,
            "connector_name": img_path.parent.name,
            "filename": img_path.name,
            "label": detect_label(img_path.stem),
            "gray_mean": gray_mean,
            "gray_std": gray_std,
        }

        if band is not None:
            band_slice = compute_band_indices(image, band)
            band_pixels = image[band_slice, :]
            record["band_mean"] = float(np.mean(band_pixels))
            record["band_std"] = float(np.std(band_pixels))

        records.append(record)
    return records


def main() -> None:
    args = parse_args()
    band: Optional[BandConfig] = None
    if args.band_rel_ymin is not None or args.band_rel_ymax is not None:
        if args.band_rel_ymin is None or args.band_rel_ymax is None:
            raise ValueError("Both --band_rel_ymin and --band_rel_ymax must be provided.")
        band = BandConfig(args.band_rel_ymin, args.band_rel_ymax)
        band.validate()

    image_paths = find_images(args.input_dir)
    print(f"Found {len(image_paths)} ROI images under '{args.input_dir}'.")

    records = build_records(image_paths, band)
    df = pd.DataFrame(records)

    output_csv: Path = args.output_csv
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_csv, index=False)

    print(f"Saved features to '{output_csv}'.")


if __name__ == "__main__":
    main()

