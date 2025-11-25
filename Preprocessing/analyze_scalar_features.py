"""
Analyze scalar features and generate plots to understand OK/KO/OCCLUSION separability.

Generates histograms and scatter plots for feature analysis, including per-connector plots.
"""

from __future__ import annotations

import argparse
import warnings
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def plot_histogram_all(
    df: pd.DataFrame, feature: str, output_path: Path
) -> None:
    """Plot histogram of a feature for all samples."""
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(df[feature].dropna(), bins=50, edgecolor="black", alpha=0.7)
    ax.set_xlabel(feature)
    ax.set_ylabel("Frequency")
    ax.set_title(f"Distribution of {feature} (All Samples)")
    ax.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Saved: {output_path.name}")


def plot_histogram_by_label(
    df: pd.DataFrame, feature: str, output_path: Path
) -> None:
    """Plot overlaid histograms of a feature grouped by label (OK/KO/OCCLUSION/PARTIAL OCCLUSION)."""
    df_labeled = df[df["label"].isin(["OK", "KO", "OCCLUSION", "PARTIAL OCCLUSION"])].copy()
    if df_labeled.empty:
        warnings.warn(f"No labeled data for {feature}, skipping histogram")
        return

    fig, ax = plt.subplots(figsize=(10, 6))

    labels = ["OK", "KO", "OCCLUSION", "PARTIAL OCCLUSION"]
    colors = {"OK": "green", "KO": "red", "OCCLUSION": "orange", "PARTIAL OCCLUSION": "yellow"}
    alphas = {"OK": 0.6, "KO": 0.6, "OCCLUSION": 0.6, "PARTIAL OCCLUSION": 0.6}

    for label in labels:
        data = df_labeled[df_labeled["label"] == label][feature].dropna()
        if len(data) > 0:
            ax.hist(
                data,
                bins=30,
                label=label,
                alpha=alphas[label],
                color=colors[label],
                edgecolor="black",
            )

    ax.set_xlabel(feature)
    ax.set_ylabel("Frequency")
    ax.set_title(f"Distribution of {feature} by Label")
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Saved: {output_path.name}")


def plot_scatter_mean_std(
    df: pd.DataFrame, output_path: Path, by_connector: Optional[str] = None
) -> None:
    """Plot scatter plot of gray_mean vs gray_std, colored by label."""
    df_labeled = df[df["label"].isin(["OK", "KO", "OCCLUSION"])].copy()
    if df_labeled.empty:
        warnings.warn("No labeled data, skipping scatter plot")
        return

    if by_connector:
        df_labeled = df_labeled[df_labeled["connector_name"] == by_connector]

    if df_labeled.empty:
        return

    fig, ax = plt.subplots(figsize=(10, 8))

    labels = ["OK", "KO", "OCCLUSION", "PARTIAL OCCLUSION"]
    colors = {"OK": "green", "KO": "red", "OCCLUSION": "orange", "PARTIAL OCCLUSION": "yellow"}
    markers = {"OK": "o", "KO": "s", "OCCLUSION": "^", "PARTIAL OCCLUSION": "d"}

    for label in labels:
        data = df_labeled[df_labeled["label"] == label]
        if len(data) > 0:
            ax.scatter(
                data["gray_mean"],
                data["gray_std"],
                label=label,
                alpha=0.6,
                s=30,
                color=colors[label],
                marker=markers[label],
                edgecolors="black",
                linewidths=0.5,
            )

    ax.set_xlabel("gray_mean")
    ax.set_ylabel("gray_std")
    title = f"gray_mean vs gray_std"
    if by_connector:
        title += f" ({by_connector})"
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Saved: {output_path.name}")


def plot_scatter_band_mean_std(df: pd.DataFrame, output_path: Path) -> None:
    """Plot scatter plot of band_mean vs band_std, colored by label."""
    if "band_mean" not in df.columns or "band_std" not in df.columns:
        warnings.warn("Band features not available, skipping band scatter plot")
        return

    df_labeled = df[df["label"].isin(["OK", "KO", "OCCLUSION", "PARTIAL OCCLUSION"])].copy()
    if df_labeled.empty:
        warnings.warn("No labeled data, skipping band scatter plot")
        return

    fig, ax = plt.subplots(figsize=(10, 8))

    labels = ["OK", "KO", "OCCLUSION", "PARTIAL OCCLUSION"]
    colors = {"OK": "green", "KO": "red", "OCCLUSION": "orange", "PARTIAL OCCLUSION": "yellow"}
    markers = {"OK": "o", "KO": "s", "OCCLUSION": "^", "PARTIAL OCCLUSION": "d"}

    for label in labels:
        data = df_labeled[df_labeled["label"] == label]
        if len(data) > 0:
            ax.scatter(
                data["band_mean"],
                data["band_std"],
                label=label,
                alpha=0.6,
                s=30,
                color=colors[label],
                marker=markers[label],
                edgecolors="black",
                linewidths=0.5,
            )

    ax.set_xlabel("band_mean")
    ax.set_ylabel("band_std")
    ax.set_title("band_mean vs band_std")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Saved: {output_path.name}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Analyze scalar features and generate plots"
    )
    parser.add_argument(
        "--input_csv",
        type=Path,
        required=True,
        help="Path to the features CSV file",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        required=True,
        help="Directory where to save plots",
    )
    parser.add_argument(
        "--by_connector",
        action="store_true",
        help="Generate per-connector plots",
    )
    args = parser.parse_args()

    # Load data
    print(f"Loading data from {args.input_csv}...")
    df = pd.read_csv(args.input_csv)
    print(f"Loaded {len(df)} samples")

    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Check label distribution
    if "label" in df.columns:
        label_counts = df["label"].value_counts()
        print(f"\nLabel distribution:")
        print(label_counts)
    else:
        warnings.warn("No 'label' column found in CSV")

    # Global histograms
    print("\nGenerating global histograms...")
    if "gray_mean" in df.columns:
        plot_histogram_all(
            df, "gray_mean", args.output_dir / "hist_gray_mean_all.png"
        )
        plot_histogram_by_label(
            df, "gray_mean", args.output_dir / "hist_gray_mean_ok_ko.png"
        )

    if "gray_std" in df.columns:
        plot_histogram_by_label(
            df, "gray_std", args.output_dir / "hist_gray_std_ok_ko.png"
        )

    # Scatter plots
    print("\nGenerating scatter plots...")
    if "gray_mean" in df.columns and "gray_std" in df.columns:
        plot_scatter_mean_std(
            df, args.output_dir / "scatter_mean_std.png"
        )

    # Band features
    if "band_mean" in df.columns:
        plot_histogram_by_label(
            df, "band_mean", args.output_dir / "hist_band_mean_ok_ko.png"
        )

    if "band_mean" in df.columns and "band_std" in df.columns:
        plot_scatter_band_mean_std(
            df, args.output_dir / "scatter_band_mean_std.png"
        )

    # Per-connector plots
    if args.by_connector:
        print("\nGenerating per-connector plots...")
        connectors = sorted(df["connector_name"].unique())
        for connector in connectors:
            plot_scatter_mean_std(
                df,
                args.output_dir / f"scatter_mean_std_{connector}.png",
                by_connector=connector,
            )

    print(f"\n✅ All plots saved to {args.output_dir}")


if __name__ == "__main__":
    main()

