"""
Alignment and illumination normalization pipeline for PCB connector dataset.

This script implements the first steps described in `Preprocessing.md`:
1. Align every raw image to a reference PCB pose using feature-matching +
   homography estimation.
2. Standardize lighting to minimize per-shot illumination differences.

Usage
-----

For raw images (first dataset):
python3 Codice/preprocess_alignment.py \
  --input-dir "Data/TOP 1" \
  --output-dir "Data/aligned_top" \
  --reference "Data/TOP 1/20251106131917_TOP.png" \
  --crop "302,288,1883,942"

For subsequent datasets (use already-aligned reference):
python3 Codice/preprocess_alignment.py \
  --input-dir "Data/TOP 2" \
  --output-dir "Data/aligned_top" \
  --reference "Data/aligned_top/20251106110559_TOP.png"

The crop parameter (optional) defines the board ROI in the aligned reference
image as xmin,ymin,xmax,ymax. When omitted, the reference is assumed to be
already aligned and cropped, so no crop is applied to the output images.

Dependencies
------------
- numpy
- opencv-python
- Pillow (only if you want to quickly inspect results elsewhere)
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence, Tuple

import cv2
import numpy as np


@dataclass(frozen=True)
class CropBox:
    xmin: int
    ymin: int
    xmax: int
    ymax: int

    @classmethod
    def parse(cls, raw: str) -> "CropBox":
        try:
            parts = [int(x.strip()) for x in raw.split(",")]
        except ValueError as exc:
            raise argparse.ArgumentTypeError(
                "Crop must be four comma-separated integers"
            ) from exc
        if len(parts) != 4:
            raise argparse.ArgumentTypeError("Crop requires exactly four integers")
        xmin, ymin, xmax, ymax = parts
        if not (xmin < xmax and ymin < ymax):
            raise argparse.ArgumentTypeError("Crop bounds must satisfy xmin<xmax, ymin<ymax")
        return cls(xmin, ymin, xmax, ymax)

    def as_slice(self) -> Tuple[slice, slice]:
        return slice(self.ymin, self.ymax), slice(self.xmin, self.xmax)


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def load_image(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"Could not read image at {path}")
    return image


def detect_homography(
    template_gray: np.ndarray,
    candidate_gray: np.ndarray,
    max_features: int = 4000,
    good_match_percent: float = 0.15,
) -> np.ndarray:
    orb = cv2.ORB_create(nfeatures=max_features, fastThreshold=5, scaleFactor=1.2)
    kp1, des1 = orb.detectAndCompute(template_gray, None)
    kp2, des2 = orb.detectAndCompute(candidate_gray, None)

    if des1 is None or des2 is None:
        raise RuntimeError("Could not extract ORB descriptors for one of the images")

    def bf_matches(cross_check: bool) -> list[cv2.DMatch]:
        matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=cross_check)
        if cross_check:
            return matcher.match(des1, des2)
        knn = matcher.knnMatch(des1, des2, k=2)
        filtered: list[cv2.DMatch] = []
        for m, n in knn:
            if m.distance < 0.75 * n.distance:
                filtered.append(m)
        return filtered

    matches = bf_matches(cross_check=True)
    if len(matches) < 15:
        matches = bf_matches(cross_check=False)
    if not matches:
        raise RuntimeError("No matches found between reference and candidate")

    matches = sorted(matches, key=lambda m: m.distance)
    keep = max(4, int(len(matches) * good_match_percent))
    matches = matches[:keep]

    src_pts = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

    H, mask = cv2.findHomography(dst_pts, src_pts, cv2.RANSAC, 4.0)
    if H is None or mask is None or mask.sum() < 8:
        raise RuntimeError("Homography estimation failed (insufficient inliers).")
    return H


def ecc_homography(
    template_gray: np.ndarray,
    candidate_gray: np.ndarray,
    iterations: int = 200,
    epsilon: float = 1e-6,
) -> np.ndarray:
    """
    Dense alignment fallback using the Enhanced Correlation Coefficient (ECC)
    optimizer. Assumes the two images already share the same resolution.
    """
    warp_matrix = np.eye(3, dtype=np.float32)
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, iterations, epsilon)
    template = template_gray.astype(np.float32) / 255.0
    candidate = candidate_gray.astype(np.float32) / 255.0
    try:
        _, warp_matrix = cv2.findTransformECC(
            template, candidate, warp_matrix, cv2.MOTION_HOMOGRAPHY, criteria, None, 5
        )
    except cv2.error as exc:
        raise RuntimeError(f"ECC optimization failed: {exc}") from exc
    return warp_matrix


def normalize_lighting(image_bgr: np.ndarray) -> np.ndarray:
    """
    Applies a combination of CLAHE (adaptive histogram equalization) and a simple
    gray-world white balance to reduce lighting differences between shots.
    """
    lab = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    l = clahe.apply(l)
    lab = cv2.merge((l, a, b))
    balanced = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

    # Gray-world white balance
    avg_b = np.mean(balanced[:, :, 0])
    avg_g = np.mean(balanced[:, :, 1])
    avg_r = np.mean(balanced[:, :, 2])
    avg_gray = (avg_b + avg_g + avg_r) / 3.0
    scale = np.array([avg_gray / avg_b, avg_gray / avg_g, avg_gray / avg_r])
    wb = balanced.astype(np.float32)
    wb *= scale
    wb = np.clip(wb, 0, 255).astype(np.uint8)
    return wb


def process_image(
    image_path: Path,
    template: np.ndarray,
    template_gray: np.ndarray,
    crop: CropBox | None,
    out_dir: Path,
    reference_already_aligned: bool = False,
) -> dict:
    image = load_image(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    try:
        H = detect_homography(template_gray, gray)
    except RuntimeError:
        H = ecc_homography(template_gray, gray)

    warped = cv2.warpPerspective(image, H, (template.shape[1], template.shape[0]))
    
    # Apply crop only if:
    # 1. Crop box is explicitly provided (for raw reference images)
    # 2. Reference is NOT already aligned (for backward compatibility)
    if crop and not reference_already_aligned:
        ys, xs = crop.as_slice()
        warped = warped[ys, xs]
    # If reference is already aligned, warped image already has correct dimensions

    normalized = normalize_lighting(warped)
    out_path = out_dir / image_path.name
    cv2.imwrite(str(out_path), normalized)

    return {
        "input": str(image_path),
        "output": str(out_path),
        "homography": H.tolist(),
    }


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="PCB alignment + illumination normalization")
    parser.add_argument("--input-dir", required=True, help="Directory with raw PCB images")
    parser.add_argument("--output-dir", required=True, help="Where aligned images will be stored")
    parser.add_argument(
        "--reference",
        required=True,
        help="Path to the reference image defining the canonical PCB pose",
    )
    parser.add_argument(
        "--crop",
        type=CropBox.parse,
        help="Optional crop window on the aligned reference (xmin,ymin,xmax,ymax)",
    )
    parser.add_argument(
        "--metadata",
        default="alignment_results.json",
        help="Filename for saved metadata inside the output directory",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    reference_path = Path(args.reference)

    ensure_dir(output_dir)

    template = load_image(reference_path)
    template_gray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
    
    # Detect if reference is already aligned:
    # If no crop is provided, assume reference is already aligned and cropped
    reference_already_aligned = args.crop is None
    if reference_already_aligned:
        print("Reference image is assumed to be already aligned (no crop provided)")
        print(f"Output images will have dimensions: {template.shape[1]}x{template.shape[0]}")

    image_paths = sorted(p for p in input_dir.glob("*.png"))
    if not image_paths:
        print(f"No .png files found inside {input_dir}", file=sys.stderr)
        return 1

    metadata: list[dict] = []
    for idx, image_path in enumerate(image_paths, start=1):
        try:
            info = process_image(
                image_path=image_path,
                template=template,
                template_gray=template_gray,
                crop=args.crop,
                out_dir=output_dir,
                reference_already_aligned=reference_already_aligned,
            )
            metadata.append(info)
            if idx % 10 == 0 or idx == len(image_paths):
                print(f"[{idx}/{len(image_paths)}] processed {image_path.name}")
        except Exception as exc:  # noqa: BLE001
            print(f"Failed on {image_path}: {exc}", file=sys.stderr)

    meta_path = output_dir / args.metadata
    meta_path.write_text(json.dumps(metadata, indent=2))
    print(f"Saved metadata for {len(metadata)} images to {meta_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

