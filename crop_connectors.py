"""
Crop the nine connector regions from aligned PCB images.

This script uses relative ROI coordinates (fractions of width/height) so it
remains valid even if we adjust the final crop size of the aligned frames.

Example
-------
python3 Codice/crop_connectors.py \
    --input-dir "Data/aligned_top" \
    --output-dir "Data/connectors" \
    --roi-config "Codice/roi_config.json" \
    --margin 8
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import cv2
import numpy as np


@dataclass(frozen=True)
class RelativeROI:
    name: str
    x_min_rel: float
    y_min_rel: float
    x_max_rel: float
    y_max_rel: float

    def to_pixel_box(self, width: int, height: int, margin: int = 0) -> tuple[int, int, int, int]:
        if not (0 <= self.x_min_rel <= 1 and 0 <= self.x_max_rel <= 1):
            raise ValueError(f"x relative coords must be in [0,1] for ROI {self.name}")
        if not (0 <= self.y_min_rel <= 1 and 0 <= self.y_max_rel <= 1):
            raise ValueError(f"y relative coords must be in [0,1] for ROI {self.name}")
        if self.x_min_rel >= self.x_max_rel or self.y_min_rel >= self.y_max_rel:
            raise ValueError(f"Invalid ROI extents for {self.name}")

        x_min = int(self.x_min_rel * width)
        y_min = int(self.y_min_rel * height)
        x_max = int(self.x_max_rel * width)
        y_max = int(self.y_max_rel * height)

        x_min = max(0, x_min - margin)
        y_min = max(0, y_min - margin)
        x_max = min(width, x_max + margin)
        y_max = min(height, y_max + margin)
        return x_min, y_min, x_max, y_max


def load_roi_config(path: Path) -> list[RelativeROI]:
    data = json.loads(path.read_text())
    rois: list[RelativeROI] = []
    for entry in data:
        rois.append(
            RelativeROI(
                name=entry["name"],
                x_min_rel=float(entry["x_min_rel"]),
                y_min_rel=float(entry["y_min_rel"]),
                x_max_rel=float(entry["x_max_rel"]),
                y_max_rel=float(entry["y_max_rel"]),
            )
        )
    if not rois:
        raise ValueError("ROI config is empty")
    return rois


def ensure_dirs(base: Path, roi_names: Iterable[str]) -> None:
    for name in roi_names:
        (base / name).mkdir(parents=True, exist_ok=True)


def crop_and_save(
    image_path: Path,
    image,
    rois: list[RelativeROI],
    output_dir: Path,
    margin: int,
    preprocess: bool = True,
) -> list[dict]:
    height, width = image.shape[:2]
    manifest_records: list[dict] = []
    for roi in rois:
        x_min, y_min, x_max, y_max = roi.to_pixel_box(width, height, margin=margin)
        crop = image[y_min:y_max, x_min:x_max]
        
        if preprocess:
            # Original behavior: Grayscale + CLAHE + Normalize
            normalized = normalize_roi(crop)
            out_img = (normalized * 255).astype(np.uint8)
        else:
            # New behavior: Raw Color Crop
            out_img = crop

        out_path = output_dir / roi.name / image_path.name
        cv2.imwrite(str(out_path), out_img)
        manifest_records.append(
            {
                "source_image": str(image_path),
                "roi_name": roi.name,
                "bbox": [x_min, y_min, x_max, y_max],
                "output": str(out_path),
            }
        )
    return manifest_records


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Crop connector ROIs from aligned PCB images.")
    parser.add_argument("--input-dir", required=True, help="Directory with aligned images.")
    parser.add_argument("--output-dir", required=True, help="Destination directory for crops.")
    parser.add_argument(
        "--roi-config",
        default="Codice/roi_config.json",
        help="JSON file containing ROI definitions.",
    )
    parser.add_argument(
        "--margin",
        type=int,
        default=10,
        help="Safety margin in pixels added around every ROI.",
    )
    parser.add_argument(
        "--metadata",
        default="connector_crops.json",
        help="Filename for the manifest saved inside output-dir.",
    )
    parser.add_argument(
        "--no-preprocess",
        action="store_true",
        help="Disable grayscale/CLAHE normalization (save raw crops).",
    )
    return parser


def normalize_roi(img: np.ndarray) -> np.ndarray:
    """
    Convert ROI to grayscale, apply CLAHE, then normalize to float32 [0, 1].
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    equalized = clahe.apply(gray)
    normalized = equalized.astype(np.float32) / 255.0
    return normalized


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    roi_path = Path(args.roi_config)

    if not input_dir.exists():
        print(f"Input dir {input_dir} does not exist", file=sys.stderr)
        return 1
    rois = load_roi_config(roi_path)
    ensure_dirs(output_dir, (roi.name for roi in rois))

    image_paths = sorted(input_dir.glob("*.png"))
    if not image_paths:
        print(f"No PNG files in {input_dir}", file=sys.stderr)
        return 1

    manifest: list[dict] = []
    for idx, image_path in enumerate(image_paths, start=1):
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None:
            print(f"Failed to read {image_path}", file=sys.stderr)
            continue

        manifest.extend(
            crop_and_save(
                image_path=image_path,
                image=image,
                rois=rois,
                output_dir=output_dir,
                margin=args.margin,
                preprocess=not args.no_preprocess,
            )
        )

        if idx % 10 == 0 or idx == len(image_paths):
            print(f"[{idx}/{len(image_paths)}] processed {image_path.name}")

    meta_path = output_dir / args.metadata
    meta_path.write_text(json.dumps(manifest, indent=2))
    print(f"Saved crop manifest with {len(manifest)} entries to {meta_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

