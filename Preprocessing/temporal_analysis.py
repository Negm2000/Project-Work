"""
Temporal Analysis of PCB Connector Quality Data

Extracts timestamp information from image filenames and generates temporal analysis plots.
Filename format: YYYYMMDDHHMMSS_TOP.png (triggered by oven sensor)

Generates:
- Production timeline analysis
- Quality trends over time
- Production rate analysis
- Time-of-day patterns
- Connector status over time
- Feature trends over time
- Illumination analysis over time
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Tuple

import cv2
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pandas import Timestamp
from tqdm import tqdm


def parse_timestamp_from_filename(filename: str) -> Timestamp | None:
    """Extract timestamp from filename format: YYYYMMDDHHMMSS_TOP.png"""
    try:
        # Remove extension and _TOP suffix
        base = filename.replace("_TOP.png", "").replace(".png", "")
        if len(base) == 14:  # YYYYMMDDHHMMSS
            year = int(base[0:4])
            month = int(base[4:6])
            day = int(base[6:8])
            hour = int(base[8:10])
            minute = int(base[10:12])
            second = int(base[12:14])
            return pd.Timestamp(year, month, day, hour, minute, second)
    except (ValueError, IndexError):
        pass
    return None


def load_and_enrich_data(csv_path: Path) -> pd.DataFrame:
    """Load CSV and add temporal columns."""
    df = pd.read_csv(csv_path)
    
    # Parse timestamps from filename
    df["timestamp"] = df["filename"].apply(parse_timestamp_from_filename)
    df = df[df["timestamp"].notna()].copy()
    
    if df.empty:
        raise ValueError("No valid timestamps found in filenames")
    
    # Add temporal features
    df["date"] = df["timestamp"].dt.date
    df["time"] = df["timestamp"].dt.time
    df["hour"] = df["timestamp"].dt.hour
    df["minute"] = df["timestamp"].dt.minute
    df["day_of_week"] = df["timestamp"].dt.day_name()
    df["is_weekend"] = df["timestamp"].dt.dayofweek >= 5
    
    # Calculate time differences (production intervals)
    df = df.sort_values("timestamp")
    df["time_since_last"] = df["timestamp"].diff().dt.total_seconds() / 60  # minutes
    
    return df


def plot_production_timeline(df: pd.DataFrame, output_dir: Path) -> None:
    """Plot production timeline showing boards over time."""
    board_timeline = df.groupby("board_id")["timestamp"].first().sort_values()
    
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.scatter(board_timeline.values, range(len(board_timeline)), 
               alpha=0.6, s=20)
    ax.set_xlabel("Time")
    ax.set_ylabel("Board Number (chronological)")
    ax.set_title("Production Timeline - Board Processing Over Time")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d %H:%M"))
    ax.xaxis.set_major_locator(mdates.HourLocator(interval=2))
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(output_dir / "production_timeline.png", dpi=150)
    plt.close()
    print(f"Saved: production_timeline.png")


def plot_production_rate(df: pd.DataFrame, output_dir: Path) -> None:
    """Plot production rate (boards per hour) over time."""
    # Count boards per hour
    hourly_counts = df.groupby(df["timestamp"].dt.floor("h"))["board_id"].nunique()
    
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.plot(hourly_counts.index, hourly_counts.values, marker="o", linewidth=2)
    ax.set_xlabel("Time")
    ax.set_ylabel("Boards per Hour")
    ax.set_title("Production Rate Over Time")
    ax.grid(True, alpha=0.3)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d %H:%M"))
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(output_dir / "production_rate.png", dpi=150)
    plt.close()
    print(f"Saved: production_rate.png")


def plot_quality_over_time(df: pd.DataFrame, output_dir: Path) -> None:
    """Plot quality (OK/KO ratio) over time."""
    df_labeled = df[df["label"].isin(["OK", "KO"])].copy()
    if df_labeled.empty:
        print("Warning: No OK/KO labels found, skipping quality over time plot")
        return
    
    # Group by hour and calculate OK/KO ratio
    hourly_quality = df_labeled.groupby(
        [df_labeled["timestamp"].dt.floor("h"), "label"]
    ).size().unstack(fill_value=0)
    
    if "OK" in hourly_quality.columns and "KO" in hourly_quality.columns:
        hourly_quality["ok_ratio"] = hourly_quality["OK"] / (
            hourly_quality["OK"] + hourly_quality["KO"]
        )
    else:
        print("Warning: Missing OK or KO labels, skipping quality plot")
        return
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), sharex=True)
    
    # OK ratio
    ax1.plot(hourly_quality.index, hourly_quality["ok_ratio"] * 100, 
             marker="o", linewidth=2, color="green")
    ax1.set_ylabel("OK Ratio (%)")
    ax1.set_title("Quality Trend Over Time - OK Ratio")
    ax1.grid(True, alpha=0.3)
    ax1.axhline(y=95, color="red", linestyle="--", alpha=0.5, label="95% threshold")
    ax1.legend()
    
    # Counts
    ax2.bar(hourly_quality.index, hourly_quality["OK"], 
            label="OK", alpha=0.7, color="green")
    ax2.bar(hourly_quality.index, hourly_quality["KO"], 
            bottom=hourly_quality["OK"], label="KO", alpha=0.7, color="red")
    ax2.set_xlabel("Time")
    ax2.set_ylabel("Count")
    ax2.set_title("OK/KO Counts Over Time")
    ax2.legend()
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d %H:%M"))
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(output_dir / "quality_over_time.png", dpi=150)
    plt.close()
    print(f"Saved: quality_over_time.png")


def plot_time_of_day_patterns(df: pd.DataFrame, output_dir: Path) -> None:
    """Plot patterns by time of day."""
    df_labeled = df[df["label"].isin(["OK", "KO"])].copy()
    if df_labeled.empty:
        print("Warning: No OK/KO labels found, skipping time-of-day patterns")
        return
    
    # Quality by hour
    hourly_quality = df_labeled.groupby("hour")["label"].apply(
        lambda x: (x == "OK").sum() / len(x) * 100
    )
    
    # Count by hour
    hourly_counts = df_labeled.groupby("hour")["label"].count()
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # OK ratio by hour
    ax1.bar(hourly_quality.index, hourly_quality.values, color="steelblue", alpha=0.7)
    ax1.set_xlabel("Hour of Day")
    ax1.set_ylabel("OK Ratio (%)")
    ax1.set_title("Quality by Hour of Day")
    ax1.set_xticks(range(24))
    ax1.grid(True, alpha=0.3, axis="y")
    
    # Production volume by hour
    ax2.bar(hourly_counts.index, hourly_counts.values, color="orange", alpha=0.7)
    ax2.set_xlabel("Hour of Day")
    ax2.set_ylabel("Number of Connectors")
    ax2.set_title("Production Volume by Hour of Day")
    ax2.set_xticks(range(24))
    ax2.grid(True, alpha=0.3, axis="y")
    
    plt.tight_layout()
    plt.savefig(output_dir / "time_of_day_patterns.png", dpi=150)
    plt.close()
    print(f"Saved: time_of_day_patterns.png")


def plot_connector_status_over_time(df: pd.DataFrame, output_dir: Path) -> None:
    """Plot status of each connector type over time."""
    df_labeled = df[df["label"].isin(["OK", "KO", "OCCLUSION"])].copy()
    if df_labeled.empty:
        print("Warning: No labels found, skipping connector status plot")
        return
    
    # Group by connector and time
    connector_timeline = df_labeled.groupby(
        [df_labeled["timestamp"].dt.floor("h"), "connector_name", "label"]
    ).size().unstack(fill_value=0)
    
    fig, axes = plt.subplots(3, 3, figsize=(18, 12), sharex=True, sharey=True)
    axes = axes.flatten()
    
    for idx, connector in enumerate(sorted(df["connector_name"].unique())):
        ax = axes[idx]
        conn_data = df_labeled[df_labeled["connector_name"] == connector]
        hourly = conn_data.groupby(
            [conn_data["timestamp"].dt.floor("h"), "label"]
        ).size().unstack(fill_value=0)
        
        if not hourly.empty:
            if "OK" in hourly.columns:
                ax.plot(hourly.index, hourly["OK"], label="OK", 
                       marker="o", markersize=3, linewidth=1.5, color="green")
            if "KO" in hourly.columns:
                ax.plot(hourly.index, hourly["KO"], label="KO", 
                       marker="s", markersize=3, linewidth=1.5, color="red")
            if "OCCLUSION" in hourly.columns:
                ax.plot(hourly.index, hourly["OCCLUSION"], label="OCCLUSION", 
                       marker="^", markersize=3, linewidth=1.5, color="orange")
        
        ax.set_title(f"{connector.upper()}")
        ax.set_ylabel("Count")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
    
    plt.suptitle("Connector Status Over Time (by Connector)", fontsize=14, y=0.995)
    plt.tight_layout()
    plt.savefig(output_dir / "connector_status_over_time.png", dpi=150)
    plt.close()
    print(f"Saved: connector_status_over_time.png")


def plot_feature_trends(df: pd.DataFrame, output_dir: Path) -> None:
    """Plot feature trends over time."""
    numeric_features = ["gray_mean", "gray_std", "band_mean", "band_std"]
    available_features = [f for f in numeric_features if f in df.columns]
    
    if not available_features:
        print("Warning: No numeric features found, skipping feature trends")
        return
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    axes = axes.flatten()
    
    for idx, feature in enumerate(available_features[:4]):
        ax = axes[idx]
        # Group by hour and calculate mean
        hourly_feature = df.groupby(df["timestamp"].dt.floor("h"))[feature].mean()
        
        ax.plot(hourly_feature.index, hourly_feature.values, 
               marker="o", linewidth=2, markersize=4)
        ax.set_xlabel("Time")
        ax.set_ylabel(feature)
        ax.set_title(f"{feature} Trend Over Time")
        ax.grid(True, alpha=0.3)
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d %H:%M"))
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
    
    plt.tight_layout()
    plt.savefig(output_dir / "feature_trends.png", dpi=150)
    plt.close()
    print(f"Saved: feature_trends.png")


def plot_production_intervals(df: pd.DataFrame, output_dir: Path) -> None:
    """Plot time intervals between board productions."""
    board_times = df.groupby("board_id")["timestamp"].first().sort_values()
    intervals = board_times.diff().dt.total_seconds() / 60  # minutes
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Histogram
    ax1.hist(intervals.dropna(), bins=50, edgecolor="black", alpha=0.7)
    ax1.set_xlabel("Time Interval (minutes)")
    ax1.set_ylabel("Frequency")
    ax1.set_title("Distribution of Production Intervals")
    ax1.grid(True, alpha=0.3, axis="y")
    
    # Over time
    ax2.plot(board_times[1:].values, intervals[1:].values, 
            marker="o", markersize=3, alpha=0.6, linewidth=1)
    ax2.set_xlabel("Time")
    ax2.set_ylabel("Interval (minutes)")
    ax2.set_title("Production Intervals Over Time")
    ax2.grid(True, alpha=0.3)
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d %H:%M"))
    plt.xticks(rotation=45)
    
    plt.tight_layout()
    plt.savefig(output_dir / "production_intervals.png", dpi=150)
    plt.close()
    print(f"Saved: production_intervals.png")


def extract_illumination_stats(image_path: Path, stage: str) -> dict | None:
    """Extract illumination statistics from a single image."""
    timestamp = parse_timestamp_from_filename(image_path.name)
    if timestamp is None:
        return None
    
    # Load image
    img = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if img is None:
        return None
    
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Calculate illumination statistics
    mean_intensity = np.mean(gray)
    std_intensity = np.std(gray)
    median_intensity = np.median(gray)
    min_intensity = np.min(gray)
    max_intensity = np.max(gray)
    
    # Calculate percentiles
    p25 = np.percentile(gray, 25)
    p75 = np.percentile(gray, 75)
    
    # Calculate spatial variability (indicator of cable presence)
    # Higher local variance suggests more cables/occlusions
    kernel_size = 15
    kernel = np.ones((kernel_size, kernel_size), np.float32) / (kernel_size * kernel_size)
    local_mean = cv2.filter2D(gray.astype(np.float32), -1, kernel)
    local_variance = cv2.filter2D((gray.astype(np.float32) - local_mean) ** 2, -1, kernel)
    mean_local_variance = np.mean(local_variance)
    
    # Calculate coefficient of variation (std/mean) - higher = more variation
    cv_coefficient = std_intensity / mean_intensity if mean_intensity > 0 else 0
    
    # Calculate range (max - min) - larger range suggests more diverse content
    intensity_range = max_intensity - min_intensity
    
    return {
        "timestamp": timestamp,
        "filename": image_path.name,
        "stage": stage,
        "mean_intensity": mean_intensity,
        "std_intensity": std_intensity,
        "median_intensity": median_intensity,
        "min_intensity": min_intensity,
        "max_intensity": max_intensity,
        "p25": p25,
        "p75": p75,
        "hour": timestamp.hour,
        "minute": timestamp.minute,
        "mean_local_variance": mean_local_variance,
        "cv_coefficient": cv_coefficient,
        "intensity_range": intensity_range,
    }


def compare_illumination_stages(
    original_dir: Path, aligned_dir: Path, connectors_dir: Path, output_dir: Path
) -> None:
    """
    Compare illumination statistics across preprocessing stages:
    1. Original images (TOP 1)
    2. Aligned images (after global normalization)
    3. Connector crops (after local normalization)
    """
    print("\n=== Comparing Illumination Across Preprocessing Stages ===")
    
    # Stage 1: Original images
    original_paths = sorted(original_dir.glob("*_TOP.png"))
    if not original_paths:
        original_paths = sorted(original_dir.glob("*.png"))
    
    # Stage 2: Aligned images
    aligned_paths = sorted(aligned_dir.glob("*.png"))
    
    # Stage 3: Connector crops (sample from one connector, e.g., conn1)
    crop_paths = sorted((connectors_dir / "conn1").glob("*.png")) if (connectors_dir / "conn1").exists() else []
    
    if not original_paths:
        print("Warning: No original images found, skipping comparison")
        return
    
    all_data = []
    
    # Extract from originals
    print(f"Extracting from {len(original_paths)} original images...")
    for img_path in tqdm(original_paths[:min(50, len(original_paths))], desc="Originals"):
        stats = extract_illumination_stats(img_path, "Original")
        if stats:
            all_data.append(stats)
    
    # Extract from aligned
    if aligned_paths:
        print(f"Extracting from {len(aligned_paths)} aligned images...")
        for img_path in tqdm(aligned_paths[:min(50, len(aligned_paths))], desc="Aligned"):
            stats = extract_illumination_stats(img_path, "Aligned")
            if stats:
                all_data.append(stats)
    
    # Extract from crops
    if crop_paths:
        print(f"Extracting from {len(crop_paths)} connector crops...")
        for img_path in tqdm(crop_paths[:min(50, len(crop_paths))], desc="Crops"):
            stats = extract_illumination_stats(img_path, "Crop")
            if stats:
                all_data.append(stats)
    
    if not all_data:
        print("Warning: No illumination data extracted, skipping comparison")
        return
    
    df = pd.DataFrame(all_data)
    
    # Summary statistics by stage
    summary = df.groupby("stage")["mean_intensity"].agg(["mean", "std", "min", "max", "count"])
    print("\n=== Illumination Statistics by Stage ===")
    print(summary.to_string())
    
    # Save summary
    summary.to_csv(output_dir / "illumination_comparison_stages.csv")
    print(f"\nSaved: illumination_comparison_stages.csv")
    
    # Plot comparison
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # 1. Mean intensity by stage (box plot)
    ax = axes[0, 0]
    stages = df["stage"].unique()
    data_by_stage = [df[df["stage"] == s]["mean_intensity"].values for s in stages]
    bp = ax.boxplot(data_by_stage, tick_labels=stages, patch_artist=True)
    colors = ["red", "blue", "green"]
    for patch, color in zip(bp["boxes"], colors[:len(bp["boxes"])]):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    ax.set_ylabel("Mean Intensity (0-255 scale)")
    ax.set_title("Mean Intensity Distribution by Preprocessing Stage")
    ax.grid(True, alpha=0.3, axis="y")
    
    # 2. Mean intensity over time by stage
    ax = axes[0, 1]
    for stage in stages:
        stage_data = df[df["stage"] == stage].sort_values("timestamp")
        ax.plot(stage_data["timestamp"], stage_data["mean_intensity"], 
               marker="o", markersize=2, alpha=0.6, label=stage, linewidth=1)
    ax.set_xlabel("Time")
    ax.set_ylabel("Mean Intensity (0-255 scale)")
    ax.set_title("Illumination Over Time by Stage")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
    
    # 3. Standard deviation by stage
    ax = axes[1, 0]
    std_by_stage = [df[df["stage"] == s]["std_intensity"].values for s in stages]
    bp = ax.boxplot(std_by_stage, tick_labels=stages, patch_artist=True)
    for patch, color in zip(bp["boxes"], colors[:len(bp["boxes"])]):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    ax.set_ylabel("Std Intensity (0-255 scale)")
    ax.set_title("Intensity Variability (Std) by Stage")
    ax.grid(True, alpha=0.3, axis="y")
    
    # 4. Histogram comparison
    ax = axes[1, 1]
    for stage, color in zip(stages, colors[:len(stages)]):
        stage_data = df[df["stage"] == stage]["mean_intensity"]
        ax.hist(stage_data, bins=30, alpha=0.6, label=stage, color=color, edgecolor="black")
    ax.set_xlabel("Mean Intensity (0-255 scale)")
    ax.set_ylabel("Frequency")
    ax.set_title("Illumination Distribution by Stage")
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")
    
    plt.tight_layout()
    plt.savefig(output_dir / "illumination_comparison_stages.png", dpi=150)
    plt.close()
    print(f"Saved: illumination_comparison_stages.png")


def create_comprehensive_illumination_summary(
    original_dir: Path, aligned_dir: Path, output_dir: Path
) -> None:
    """
    Create comprehensive summary plots comparing:
    1. Morning vs Afternoon in TOP 1 (original images)
    2. Morning vs Afternoon in aligned images (after processing)
    3. Comparison between TOP 1 and aligned
    4. Cable influence on both stages
    """
    print("\n=== Creating Comprehensive Illumination Summary ===")
    
    # Load data from both stages
    original_paths = sorted(original_dir.glob("*_TOP.png"))
    if not original_paths:
        original_paths = sorted(original_dir.glob("*.png"))
    
    aligned_paths = sorted(aligned_dir.glob("*.png"))
    
    if not original_paths:
        print("Warning: No original images found")
        return
    
    print(f"Processing {len(original_paths)} original images and {len(aligned_paths)} aligned images...")
    
    # Extract statistics
    original_data = []
    for img_path in tqdm(original_paths, desc="Originals"):
        stats = extract_illumination_stats(img_path, "Original")
        if stats:
            original_data.append(stats)
    
    aligned_data = []
    if aligned_paths:
        for img_path in tqdm(aligned_paths, desc="Aligned"):
            stats = extract_illumination_stats(img_path, "Aligned")
            if stats:
                aligned_data.append(stats)
    
    if not original_data:
        print("Warning: No data extracted")
        return
    
    df_original = pd.DataFrame(original_data)
    df_aligned = pd.DataFrame(aligned_data) if aligned_data else pd.DataFrame()
    
    # Define time periods
    def get_period(hour: int) -> str:
        if 6 <= hour <= 11:
            return "Morning"
        elif 12 <= hour <= 17:
            return "Afternoon"
        else:
            return "Other"
    
    df_original["period"] = df_original["hour"].apply(get_period)
    if not df_aligned.empty:
        df_aligned["period"] = df_aligned["hour"].apply(get_period)
    
    # Filter to Morning and Afternoon only
    df_orig_morning = df_original[df_original["period"] == "Morning"]
    df_orig_afternoon = df_original[df_original["period"] == "Afternoon"]
    
    df_align_morning = df_aligned[df_aligned["period"] == "Morning"] if not df_aligned.empty else pd.DataFrame()
    df_align_afternoon = df_aligned[df_aligned["period"] == "Afternoon"] if not df_aligned.empty else pd.DataFrame()
    
    # Create comprehensive summary plot
    fig = plt.figure(figsize=(20, 14))
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
    
    # === ROW 1: Morning vs Afternoon Comparison ===
    
    # 1. TOP 1: Morning vs Afternoon (Box plot)
    ax1 = fig.add_subplot(gs[0, 0])
    data_orig = [df_orig_morning["mean_intensity"].values, df_orig_afternoon["mean_intensity"].values]
    bp1 = ax1.boxplot(data_orig, tick_labels=["Morning\n(TOP 1)", "Afternoon\n(TOP 1)"], 
                      patch_artist=True)
    bp1["boxes"][0].set_facecolor("orange")
    bp1["boxes"][1].set_facecolor("yellow")
    for box in bp1["boxes"]:
        box.set_alpha(0.7)
    ax1.set_ylabel("Mean Intensity (0-255)")
    ax1.set_title("TOP 1: Morning vs Afternoon")
    ax1.grid(True, alpha=0.3, axis="y")
    
    # Add statistics text
    if not df_orig_morning.empty and not df_orig_afternoon.empty:
        diff_orig = df_orig_afternoon["mean_intensity"].mean() - df_orig_morning["mean_intensity"].mean()
        ax1.text(0.5, 0.95, f"Δ = {diff_orig:.2f} points", 
                transform=ax1.transAxes, ha="center", va="top",
                bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))
    
    # 2. Aligned: Morning vs Afternoon (Box plot)
    ax2 = fig.add_subplot(gs[0, 1])
    if not df_align_morning.empty and not df_align_afternoon.empty:
        data_align = [df_align_morning["mean_intensity"].values, df_align_afternoon["mean_intensity"].values]
        bp2 = ax2.boxplot(data_align, tick_labels=["Morning\n(Aligned)", "Afternoon\n(Aligned)"], 
                         patch_artist=True)
        bp2["boxes"][0].set_facecolor("orange")
        bp2["boxes"][1].set_facecolor("yellow")
        for box in bp2["boxes"]:
            box.set_alpha(0.7)
        diff_align = df_align_afternoon["mean_intensity"].mean() - df_align_morning["mean_intensity"].mean()
        ax2.text(0.5, 0.95, f"Δ = {diff_align:.2f} points", 
                transform=ax2.transAxes, ha="center", va="top",
                bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))
    else:
        ax2.text(0.5, 0.5, "No aligned data", ha="center", va="center", transform=ax2.transAxes)
    ax2.set_ylabel("Mean Intensity (0-255)")
    ax2.set_title("Aligned: Morning vs Afternoon")
    ax2.grid(True, alpha=0.3, axis="y")
    
    # 3. Comparison: TOP 1 vs Aligned (all periods combined)
    ax3 = fig.add_subplot(gs[0, 2])
    if not df_aligned.empty:
        data_compare = [df_original["mean_intensity"].values, df_aligned["mean_intensity"].values]
        bp3 = ax3.boxplot(data_compare, tick_labels=["TOP 1\n(Original)", "Aligned\n(Processed)"], 
                          patch_artist=True)
        bp3["boxes"][0].set_facecolor("red")
        bp3["boxes"][1].set_facecolor("blue")
        for box in bp3["boxes"]:
            box.set_alpha(0.7)
        diff_stages = df_aligned["mean_intensity"].mean() - df_original["mean_intensity"].mean()
        ax3.text(0.5, 0.95, f"Δ = {diff_stages:.2f} points", 
                transform=ax3.transAxes, ha="center", va="top",
                bbox=dict(boxstyle="round", facecolor="lightblue", alpha=0.5))
    else:
        ax3.text(0.5, 0.5, "No aligned data", ha="center", va="center", transform=ax3.transAxes)
    ax3.set_ylabel("Mean Intensity (0-255)")
    ax3.set_title("TOP 1 vs Aligned (Overall)")
    ax3.grid(True, alpha=0.3, axis="y")
    
    # === ROW 2: Time Series with Cable Influence ===
    
    # 4. TOP 1: Intensity over time (colored by cable presence)
    ax4 = fig.add_subplot(gs[1, 0])
    scatter1 = ax4.scatter(df_original["timestamp"], df_original["mean_intensity"],
                          c=df_original["mean_local_variance"], cmap="hot",
                          s=40, alpha=0.7, edgecolors="black", linewidth=0.5)
    ax4.set_xlabel("Time")
    ax4.set_ylabel("Mean Intensity (0-255)")
    ax4.set_title("TOP 1: Illumination Over Time\n(Colored by Cable Presence)")
    ax4.grid(True, alpha=0.3)
    ax4.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
    plt.setp(ax4.xaxis.get_majorticklabels(), rotation=45)
    cbar1 = plt.colorbar(scatter1, ax=ax4)
    cbar1.set_label("Local Variance\n(Cable Indicator)")
    
    # 5. Aligned: Intensity over time (colored by cable presence)
    ax5 = fig.add_subplot(gs[1, 1])
    if not df_aligned.empty:
        scatter2 = ax5.scatter(df_aligned["timestamp"], df_aligned["mean_intensity"],
                               c=df_aligned["mean_local_variance"], cmap="hot",
                               s=40, alpha=0.7, edgecolors="black", linewidth=0.5)
        ax5.set_xlabel("Time")
        ax5.set_ylabel("Mean Intensity (0-255)")
        ax5.set_title("Aligned: Illumination Over Time\n(Colored by Cable Presence)")
        ax5.grid(True, alpha=0.3)
        ax5.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
        plt.setp(ax5.xaxis.get_majorticklabels(), rotation=45)
        cbar2 = plt.colorbar(scatter2, ax=ax5)
        cbar2.set_label("Local Variance\n(Cable Indicator)")
    else:
        ax5.text(0.5, 0.5, "No aligned data", ha="center", va="center", transform=ax5.transAxes)
    
    # 6. Cable presence over time (both stages)
    ax6 = fig.add_subplot(gs[1, 2])
    ax6.plot(df_original["timestamp"], df_original["mean_local_variance"],
            marker="o", markersize=3, alpha=0.6, label="TOP 1", linewidth=1.5, color="red")
    if not df_aligned.empty:
        ax6.plot(df_aligned["timestamp"], df_aligned["mean_local_variance"],
                marker="s", markersize=3, alpha=0.6, label="Aligned", linewidth=1.5, color="blue")
    ax6.set_xlabel("Time")
    ax6.set_ylabel("Mean Local Variance")
    ax6.set_title("Cable Presence Indicator Over Time")
    ax6.legend()
    ax6.grid(True, alpha=0.3)
    ax6.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
    plt.setp(ax6.xaxis.get_majorticklabels(), rotation=45)
    
    # === ROW 3: Cable Impact Analysis ===
    
    # 7. TOP 1: Cable impact on morning/afternoon
    ax7 = fig.add_subplot(gs[2, 0])
    median_var_orig = df_original["mean_local_variance"].median()
    orig_low_cables = df_original[df_original["mean_local_variance"] < median_var_orig]
    orig_high_cables = df_original[df_original["mean_local_variance"] >= median_var_orig]
    
    orig_morning_low = orig_low_cables[orig_low_cables["period"] == "Morning"]["mean_intensity"]
    orig_morning_high = orig_high_cables[orig_high_cables["period"] == "Morning"]["mean_intensity"]
    orig_afternoon_low = orig_low_cables[orig_low_cables["period"] == "Afternoon"]["mean_intensity"]
    orig_afternoon_high = orig_high_cables[orig_high_cables["period"] == "Afternoon"]["mean_intensity"]
    
    data_cable_orig = []
    labels_cable_orig = []
    if not orig_morning_low.empty:
        data_cable_orig.append(orig_morning_low.values)
        labels_cable_orig.append("Morn\nLow Cables")
    if not orig_morning_high.empty:
        data_cable_orig.append(orig_morning_high.values)
        labels_cable_orig.append("Morn\nHigh Cables")
    if not orig_afternoon_low.empty:
        data_cable_orig.append(orig_afternoon_low.values)
        labels_cable_orig.append("Aft\nLow Cables")
    if not orig_afternoon_high.empty:
        data_cable_orig.append(orig_afternoon_high.values)
        labels_cable_orig.append("Aft\nHigh Cables")
    
    if data_cable_orig:
        bp7 = ax7.boxplot(data_cable_orig, tick_labels=labels_cable_orig, patch_artist=True)
        colors7 = ["lightblue", "orange", "lightgreen", "coral"]
        for patch, color in zip(bp7["boxes"], colors7[:len(bp7["boxes"])]):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
    ax7.set_ylabel("Mean Intensity (0-255)")
    ax7.set_title("TOP 1: Cable Impact by Period")
    ax7.grid(True, alpha=0.3, axis="y")
    
    # 8. Aligned: Cable impact on morning/afternoon
    ax8 = fig.add_subplot(gs[2, 1])
    if not df_aligned.empty:
        median_var_align = df_aligned["mean_local_variance"].median()
        align_low_cables = df_aligned[df_aligned["mean_local_variance"] < median_var_align]
        align_high_cables = df_aligned[df_aligned["mean_local_variance"] >= median_var_align]
        
        align_morning_low = align_low_cables[align_low_cables["period"] == "Morning"]["mean_intensity"]
        align_morning_high = align_high_cables[align_high_cables["period"] == "Morning"]["mean_intensity"]
        align_afternoon_low = align_low_cables[align_low_cables["period"] == "Afternoon"]["mean_intensity"]
        align_afternoon_high = align_high_cables[align_high_cables["period"] == "Afternoon"]["mean_intensity"]
        
        data_cable_align = []
        labels_cable_align = []
        if not align_morning_low.empty:
            data_cable_align.append(align_morning_low.values)
            labels_cable_align.append("Morn\nLow Cables")
        if not align_morning_high.empty:
            data_cable_align.append(align_morning_high.values)
            labels_cable_align.append("Morn\nHigh Cables")
        if not align_afternoon_low.empty:
            data_cable_align.append(align_afternoon_low.values)
            labels_cable_align.append("Aft\nLow Cables")
        if not align_afternoon_high.empty:
            data_cable_align.append(align_afternoon_high.values)
            labels_cable_align.append("Aft\nHigh Cables")
        
        if data_cable_align:
            bp8 = ax8.boxplot(data_cable_align, tick_labels=labels_cable_align, patch_artist=True)
            colors8 = ["lightblue", "orange", "lightgreen", "coral"]
            for patch, color in zip(bp8["boxes"], colors8[:len(bp8["boxes"])]):
                patch.set_facecolor(color)
                patch.set_alpha(0.7)
    else:
        ax8.text(0.5, 0.5, "No aligned data", ha="center", va="center", transform=ax8.transAxes)
    ax8.set_ylabel("Mean Intensity (0-255)")
    ax8.set_title("Aligned: Cable Impact by Period")
    ax8.grid(True, alpha=0.3, axis="y")
    
    # 9. Summary statistics table
    ax9 = fig.add_subplot(gs[2, 2])
    ax9.axis("off")
    
    # Create summary table
    summary_data = []
    
    # TOP 1 statistics
    if not df_orig_morning.empty:
        summary_data.append(["TOP 1 - Morning", f"{df_orig_morning['mean_intensity'].mean():.2f}",
                           f"±{df_orig_morning['mean_intensity'].std():.2f}", len(df_orig_morning)])
    if not df_orig_afternoon.empty:
        summary_data.append(["TOP 1 - Afternoon", f"{df_orig_afternoon['mean_intensity'].mean():.2f}",
                           f"±{df_orig_afternoon['mean_intensity'].std():.2f}", len(df_orig_afternoon)])
    if not df_orig_morning.empty and not df_orig_afternoon.empty:
        diff_orig_val = df_orig_afternoon["mean_intensity"].mean() - df_orig_morning["mean_intensity"].mean()
        summary_data.append(["TOP 1 - Difference", f"{diff_orig_val:.2f}", "", ""])
    
    # Aligned statistics
    if not df_align_morning.empty:
        summary_data.append(["Aligned - Morning", f"{df_align_morning['mean_intensity'].mean():.2f}",
                           f"±{df_align_morning['mean_intensity'].std():.2f}", len(df_align_morning)])
    if not df_align_afternoon.empty:
        summary_data.append(["Aligned - Afternoon", f"{df_align_afternoon['mean_intensity'].mean():.2f}",
                           f"±{df_align_afternoon['mean_intensity'].std():.2f}", len(df_align_afternoon)])
    if not df_align_morning.empty and not df_align_afternoon.empty:
        diff_align_val = df_align_afternoon["mean_intensity"].mean() - df_align_morning["mean_intensity"].mean()
        summary_data.append(["Aligned - Difference", f"{diff_align_val:.2f}", "", ""])
    
    # Cable correlations
    corr_orig = df_original["mean_local_variance"].corr(df_original["mean_intensity"])
    summary_data.append(["TOP 1 - Cable Corr", f"{corr_orig:.3f}", "", ""])
    if not df_aligned.empty:
        corr_align = df_aligned["mean_local_variance"].corr(df_aligned["mean_intensity"])
        summary_data.append(["Aligned - Cable Corr", f"{corr_align:.3f}", "", ""])
    
    if summary_data:
        table = ax9.table(cellText=summary_data,
                         colLabels=["Metric", "Mean", "Std", "Count"],
                         cellLoc="center",
                         loc="center",
                         bbox=[0, 0, 1, 1])
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1, 2)
        for i in range(len(summary_data) + 1):
            for j in range(4):
                cell = table[(i, j)]
                if i == 0:  # Header
                    cell.set_facecolor("#4472C4")
                    cell.set_text_props(weight="bold", color="white")
                else:
                    if i % 2 == 0:
                        cell.set_facecolor("#D9E1F2")
                    else:
                        cell.set_facecolor("white")
        ax9.set_title("Summary Statistics", pad=20, fontsize=12, weight="bold")
    
    plt.suptitle("Comprehensive Illumination Analysis: TOP 1 vs Aligned\nMorning/Afternoon Comparison & Cable Influence", 
                fontsize=16, weight="bold", y=0.995)
    
    plt.savefig(output_dir / "illumination_comprehensive_summary.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: illumination_comprehensive_summary.png")
    
    # Save detailed statistics
    stats_summary = []
    if not df_orig_morning.empty:
        stats_summary.append({
            "Stage": "TOP 1",
            "Period": "Morning",
            "Mean": f"{df_orig_morning['mean_intensity'].mean():.2f}",
            "Std": f"{df_orig_morning['mean_intensity'].std():.2f}",
            "Count": len(df_orig_morning)
        })
    if not df_orig_afternoon.empty:
        stats_summary.append({
            "Stage": "TOP 1",
            "Period": "Afternoon",
            "Mean": f"{df_orig_afternoon['mean_intensity'].mean():.2f}",
            "Std": f"{df_orig_afternoon['mean_intensity'].std():.2f}",
            "Count": len(df_orig_afternoon)
        })
    if not df_align_morning.empty:
        stats_summary.append({
            "Stage": "Aligned",
            "Period": "Morning",
            "Mean": f"{df_align_morning['mean_intensity'].mean():.2f}",
            "Std": f"{df_align_morning['mean_intensity'].std():.2f}",
            "Count": len(df_align_morning)
        })
    if not df_align_afternoon.empty:
        stats_summary.append({
            "Stage": "Aligned",
            "Period": "Afternoon",
            "Mean": f"{df_align_afternoon['mean_intensity'].mean():.2f}",
            "Std": f"{df_align_afternoon['mean_intensity'].std():.2f}",
            "Count": len(df_align_afternoon)
        })
    
    if stats_summary:
        stats_df = pd.DataFrame(stats_summary)
        stats_df.to_csv(output_dir / "illumination_comprehensive_stats.csv", index=False)
        print(f"Saved: illumination_comprehensive_stats.csv")
        
        print("\n=== Comprehensive Summary Statistics ===")
        print(stats_df.to_string(index=False))


def analyze_cable_impact_on_illumination(
    original_images_dir: Path, output_dir: Path
) -> None:
    """
    Analyze if the presence of cables (occupying more space) affects illumination statistics.
    
    Uses spatial variability metrics to detect cable presence and correlates with
    illumination measurements over time.
    """
    print("\n=== Analyzing Cable Impact on Illumination ===")
    
    image_paths = sorted(original_images_dir.glob("*_TOP.png"))
    if not image_paths:
        image_paths = sorted(original_images_dir.glob("*.png"))
    
    if not image_paths:
        print("Warning: No images found, skipping cable impact analysis")
        return
    
    print(f"Analyzing {len(image_paths)} images for cable presence indicators...")
    
    cable_data = []
    
    for image_path in tqdm(image_paths, desc="Analyzing cable impact"):
        stats = extract_illumination_stats(image_path, "Original")
        if stats:
            cable_data.append(stats)
    
    if not cable_data:
        print("Warning: No data extracted, skipping analysis")
        return
    
    df = pd.DataFrame(cable_data)
    df = df.sort_values("timestamp")
    
    # Correlation analysis
    print("\n=== Correlation Analysis ===")
    correlations = df[["mean_intensity", "std_intensity", "mean_local_variance", 
                       "cv_coefficient", "intensity_range"]].corr()
    print(correlations.to_string())
    
    # Save correlations
    correlations.to_csv(output_dir / "illumination_cable_correlations.csv")
    print(f"\nSaved: illumination_cable_correlations.csv")
    
    # Plot analysis
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # 1. Mean intensity vs Local variance (cable indicator)
    ax = axes[0, 0]
    ax.scatter(df["mean_local_variance"], df["mean_intensity"], 
              alpha=0.6, s=30, c=df["hour"], cmap="viridis")
    ax.set_xlabel("Mean Local Variance (Cable Presence Indicator)")
    ax.set_ylabel("Mean Intensity (0-255)")
    ax.set_title("Illumination vs Cable Presence\n(Higher variance = more cables)")
    ax.grid(True, alpha=0.3)
    cbar = plt.colorbar(ax.collections[0], ax=ax)
    cbar.set_label("Hour of Day")
    
    # 2. Mean intensity vs Coefficient of Variation
    ax = axes[0, 1]
    ax.scatter(df["cv_coefficient"], df["mean_intensity"], 
              alpha=0.6, s=30, c=df["hour"], cmap="viridis")
    ax.set_xlabel("Coefficient of Variation (Std/Mean)")
    ax.set_ylabel("Mean Intensity (0-255)")
    ax.set_title("Illumination vs Spatial Variability")
    ax.grid(True, alpha=0.3)
    cbar = plt.colorbar(ax.collections[0], ax=ax)
    cbar.set_label("Hour of Day")
    
    # 3. Local variance over time
    ax = axes[0, 2]
    ax.plot(df["timestamp"], df["mean_local_variance"], 
           marker="o", markersize=3, alpha=0.6, linewidth=1)
    ax.set_xlabel("Time")
    ax.set_ylabel("Mean Local Variance")
    ax.set_title("Cable Presence Indicator Over Time\n(Higher = more cables visible)")
    ax.grid(True, alpha=0.3)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
    
    # 4. Mean intensity over time (colored by local variance)
    ax = axes[1, 0]
    scatter = ax.scatter(df["timestamp"], df["mean_intensity"], 
                        c=df["mean_local_variance"], cmap="hot", 
                        s=30, alpha=0.7, edgecolors="black", linewidth=0.5)
    ax.set_xlabel("Time")
    ax.set_ylabel("Mean Intensity (0-255)")
    ax.set_title("Illumination Over Time\n(Colored by Cable Presence)")
    ax.grid(True, alpha=0.3)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label("Local Variance (Cable Indicator)")
    
    # 5. Box plot: Mean intensity by high/low cable presence
    ax = axes[1, 1]
    median_variance = df["mean_local_variance"].median()
    high_cables = df[df["mean_local_variance"] >= median_variance]["mean_intensity"]
    low_cables = df[df["mean_local_variance"] < median_variance]["mean_intensity"]
    
    bp = ax.boxplot([low_cables, high_cables], 
                    tick_labels=["Low Cable\nPresence", "High Cable\nPresence"],
                    patch_artist=True)
    colors_box = ["lightblue", "orange"]
    for patch, color in zip(bp["boxes"], colors_box):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    ax.set_ylabel("Mean Intensity (0-255)")
    ax.set_title("Illumination: Low vs High Cable Presence")
    ax.grid(True, alpha=0.3, axis="y")
    
    # 6. Hourly pattern of cable presence
    ax = axes[1, 2]
    hourly_cables = df.groupby("hour")["mean_local_variance"].agg(["mean", "std"])
    ax.bar(hourly_cables.index, hourly_cables["mean"], 
           yerr=hourly_cables["std"], capsize=5, alpha=0.7, 
           color="coral", edgecolor="black")
    ax.set_xlabel("Hour of Day")
    ax.set_ylabel("Mean Local Variance")
    ax.set_title("Cable Presence by Hour of Day")
    ax.set_xticks(range(24))
    ax.grid(True, alpha=0.3, axis="y")
    
    plt.tight_layout()
    plt.savefig(output_dir / "cable_impact_on_illumination.png", dpi=150)
    plt.close()
    print(f"Saved: cable_impact_on_illumination.png")
    
    # Statistical summary
    print("\n=== Statistical Summary ===")
    print(f"Median local variance: {df['mean_local_variance'].median():.2f}")
    print(f"Mean intensity (low cables): {low_cables.mean():.2f} ± {low_cables.std():.2f}")
    print(f"Mean intensity (high cables): {high_cables.mean():.2f} ± {high_cables.std():.2f}")
    print(f"Difference: {abs(high_cables.mean() - low_cables.mean()):.2f} points")
    
    # Correlation coefficient
    corr_coef = df["mean_local_variance"].corr(df["mean_intensity"])
    print(f"\nCorrelation (Local Variance vs Mean Intensity): {corr_coef:.3f}")
    
    if abs(corr_coef) > 0.3:
        print("⚠️  STRONG correlation detected - cables may be affecting illumination!")
    elif abs(corr_coef) > 0.15:
        print("⚠️  Moderate correlation detected - cables may have some influence")
    else:
        print("✓ Weak correlation - cables don't seem to strongly affect illumination")


def analyze_illumination_over_time(
    original_images_dir: Path, output_dir: Path
) -> None:
    """
    Analyze illumination statistics over time from ORIGINAL images (before normalization).
    
    This analysis uses the raw images to detect actual lighting differences between
    morning and evening shots, before any normalization is applied.
    
    Scale: Intensity values are in 0-255 range (8-bit grayscale).
    """
    if not original_images_dir.exists():
        print(f"Warning: Original images directory {original_images_dir} not found, skipping illumination analysis")
        return
    
    image_paths = sorted(original_images_dir.glob("*_TOP.png"))
    if not image_paths:
        # Try without _TOP suffix
        image_paths = sorted(original_images_dir.glob("*.png"))
    
    if not image_paths:
        print(f"Warning: No PNG files found in {original_images_dir}, skipping illumination analysis")
        return
    
    print(f"\nAnalyzing illumination from {len(image_paths)} ORIGINAL images (before normalization)...")
    print("Note: Intensity values are in 0-255 scale (8-bit grayscale)")
    
    illumination_data = []
    
    for image_path in tqdm(image_paths, desc="Processing images"):
        stats = extract_illumination_stats(image_path, "Original")
        if stats:
            illumination_data.append(stats)
    
    if not illumination_data:
        print("Warning: No valid illumination data extracted, skipping plots")
        return
    
    illum_df = pd.DataFrame(illumination_data)
    illum_df = illum_df.sort_values("timestamp")
    
    # Plot 1: Mean intensity over time
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.plot(illum_df["timestamp"], illum_df["mean_intensity"], 
            marker="o", markersize=3, alpha=0.6, linewidth=1)
    ax.set_xlabel("Time")
    ax.set_ylabel("Mean Intensity")
    ax.set_title("Illumination Over Time - Mean Intensity (Original Images, Before Normalization)")
    ax.grid(True, alpha=0.3)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d %H:%M"))
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(output_dir / "illumination_over_time.png", dpi=150)
    plt.close()
    print(f"Saved: illumination_over_time.png")
    
    # Plot 2: Mean intensity by hour of day
    hourly_illum = illum_df.groupby("hour")["mean_intensity"].agg(["mean", "std", "min", "max"])
    
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.bar(hourly_illum.index, hourly_illum["mean"], 
           yerr=hourly_illum["std"], capsize=5, alpha=0.7, color="steelblue", 
           edgecolor="black", label="Mean ± Std")
    ax.plot(hourly_illum.index, hourly_illum["min"], 
           marker="v", markersize=5, color="red", linestyle="--", label="Min")
    ax.plot(hourly_illum.index, hourly_illum["max"], 
           marker="^", markersize=5, color="green", linestyle="--", label="Max")
    ax.set_xlabel("Hour of Day")
    ax.set_ylabel("Intensity")
    ax.set_title("Illumination by Hour of Day - Mean, Min, Max (Original Images)")
    ax.set_xticks(range(24))
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    plt.savefig(output_dir / "illumination_by_hour.png", dpi=150)
    plt.close()
    print(f"Saved: illumination_by_hour.png")
    
    # Plot 3: Comparison morning vs evening
    morning = illum_df[illum_df["hour"].between(6, 11)]  # 6-11 AM
    afternoon = illum_df[illum_df["hour"].between(12, 17)]  # 12-17 PM
    evening = illum_df[illum_df["hour"].between(18, 23)]  # 18-23 PM
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Histogram comparison
    if not morning.empty:
        ax1.hist(morning["mean_intensity"], bins=30, alpha=0.6, label="Morning (6-11h)", 
                color="orange", edgecolor="black")
    if not afternoon.empty:
        ax1.hist(afternoon["mean_intensity"], bins=30, alpha=0.6, label="Afternoon (12-17h)", 
                color="yellow", edgecolor="black")
    if not evening.empty:
        ax1.hist(evening["mean_intensity"], bins=30, alpha=0.6, label="Evening (18-23h)", 
                color="purple", edgecolor="black")
    ax1.set_xlabel("Mean Intensity")
    ax1.set_ylabel("Frequency")
    ax1.set_title("Illumination Distribution: Morning vs Afternoon vs Evening (Original Images)")
    ax1.legend()
    ax1.grid(True, alpha=0.3, axis="y")
    
    # Box plot comparison
    data_to_plot = []
    labels = []
    if not morning.empty:
        data_to_plot.append(morning["mean_intensity"].values)
        labels.append("Morning\n(6-11h)")
    if not afternoon.empty:
        data_to_plot.append(afternoon["mean_intensity"].values)
        labels.append("Afternoon\n(12-17h)")
    if not evening.empty:
        data_to_plot.append(evening["mean_intensity"].values)
        labels.append("Evening\n(18-23h)")
    
    if data_to_plot:
        bp = ax2.boxplot(data_to_plot, tick_labels=labels, patch_artist=True)
        colors = ["orange", "yellow", "purple"]
        for patch, color in zip(bp["boxes"], colors[:len(bp["boxes"])]):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        ax2.set_ylabel("Mean Intensity")
        ax2.set_title("Illumination Comparison: Morning vs Afternoon vs Evening (Original Images)")
        ax2.grid(True, alpha=0.3, axis="y")
    
    plt.tight_layout()
    plt.savefig(output_dir / "illumination_morning_evening.png", dpi=150)
    plt.close()
    print(f"Saved: illumination_morning_evening.png")
    
    # Save statistics
    stats = []
    if not morning.empty:
        stats.append({"Period": "Morning (6-11h)", 
                     "Mean": f"{morning['mean_intensity'].mean():.2f}",
                     "Std": f"{morning['mean_intensity'].std():.2f}",
                     "Min": f"{morning['mean_intensity'].min():.2f}",
                     "Max": f"{morning['mean_intensity'].max():.2f}",
                     "Count": len(morning)})
    if not afternoon.empty:
        stats.append({"Period": "Afternoon (12-17h)", 
                     "Mean": f"{afternoon['mean_intensity'].mean():.2f}",
                     "Std": f"{afternoon['mean_intensity'].std():.2f}",
                     "Min": f"{afternoon['mean_intensity'].min():.2f}",
                     "Max": f"{afternoon['mean_intensity'].max():.2f}",
                     "Count": len(afternoon)})
    if not evening.empty:
        stats.append({"Period": "Evening (18-23h)", 
                     "Mean": f"{evening['mean_intensity'].mean():.2f}",
                     "Std": f"{evening['mean_intensity'].std():.2f}",
                     "Min": f"{evening['mean_intensity'].min():.2f}",
                     "Max": f"{evening['mean_intensity'].max():.2f}",
                     "Count": len(evening)})
    
    if stats:
        stats_df = pd.DataFrame(stats)
        stats_df.to_csv(output_dir / "illumination_stats.csv", index=False)
        print(f"Saved: illumination_stats.csv")
        
        # Print summary
        print("\n=== Illumination Analysis Summary (Original Images, Before Normalization) ===")
        print(stats_df.to_string(index=False))
        print("\nNote: These statistics show the ACTUAL lighting differences before normalization.")
        print("If differences are large, it confirms the importance of illumination normalization.")


def generate_summary_statistics(df: pd.DataFrame, output_dir: Path) -> None:
    """Generate summary statistics CSV."""
    stats = []
    
    # Time span
    time_span = df["timestamp"].max() - df["timestamp"].min()
    stats.append({"Metric": "Total Time Span", 
                 "Value": f"{time_span.days} days, {time_span.seconds // 3600} hours"})
    
    # Total boards
    stats.append({"Metric": "Total Boards", "Value": df["board_id"].nunique()})
    
    # Production rate
    avg_interval = df.groupby("board_id")["timestamp"].first().diff().dt.total_seconds().mean() / 60
    stats.append({"Metric": "Average Interval (minutes)", "Value": f"{avg_interval:.2f}"})
    
    # Quality stats
    df_labeled = df[df["label"].isin(["OK", "KO"])].copy()
    if not df_labeled.empty:
        ok_ratio = (df_labeled["label"] == "OK").sum() / len(df_labeled) * 100
        stats.append({"Metric": "Overall OK Ratio (%)", "Value": f"{ok_ratio:.2f}"})
    
    # Time patterns
    stats.append({"Metric": "Peak Production Hour", 
                 "Value": str(df.groupby("hour")["board_id"].nunique().idxmax())})
    
    stats_df = pd.DataFrame(stats)
    stats_df.to_csv(output_dir / "temporal_summary_stats.csv", index=False)
    print(f"Saved: temporal_summary_stats.csv")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Temporal analysis of PCB connector quality data"
    )
    parser.add_argument(
        "--input_csv",
        type=Path,
        default=Path("features_labeled.csv"),
        help="Input CSV with labeled features",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("results/temporal_analysis"),
        help="Output directory for plots",
    )
    parser.add_argument(
        "--original_images_dir",
        type=Path,
        default=Path("Data/TOP 1"),
        help="Directory with ORIGINAL images (before normalization) for illumination analysis",
    )
    parser.add_argument(
        "--aligned_images_dir",
        type=Path,
        default=Path("Data/aligned_top"),
        help="Directory with aligned images (after global normalization)",
    )
    parser.add_argument(
        "--connectors_dir",
        type=Path,
        default=Path("Data/connectors"),
        help="Directory with connector crops (after local normalization)",
    )
    args = parser.parse_args()
    
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    print("Loading and enriching data with temporal information...")
    df = load_and_enrich_data(args.input_csv)
    print(f"Loaded {len(df)} samples from {df['board_id'].nunique()} boards")
    print(f"Time span: {df['timestamp'].min()} to {df['timestamp'].max()}")
    
    print("\nGenerating temporal analysis plots...")
    plot_production_timeline(df, args.output_dir)
    plot_production_rate(df, args.output_dir)
    plot_quality_over_time(df, args.output_dir)
    plot_time_of_day_patterns(df, args.output_dir)
    plot_connector_status_over_time(df, args.output_dir)
    plot_feature_trends(df, args.output_dir)
    plot_production_intervals(df, args.output_dir)
    generate_summary_statistics(df, args.output_dir)
    
    # Illumination analysis (using ORIGINAL images to see real lighting differences)
    analyze_illumination_over_time(args.original_images_dir, args.output_dir)
    
    # Compare illumination across preprocessing stages
    compare_illumination_stages(
        args.original_images_dir,
        args.aligned_images_dir,
        args.connectors_dir,
        args.output_dir,
    )
    
    # Analyze cable impact on illumination
    analyze_cable_impact_on_illumination(args.original_images_dir, args.output_dir)
    
    # Create comprehensive summary (TOP 1 vs Aligned, Morning/Afternoon, Cable influence)
    create_comprehensive_illumination_summary(
        args.original_images_dir,
        args.aligned_images_dir,
        args.output_dir,
    )
    
    print(f"\n✅ All plots saved to {args.output_dir}")


if __name__ == "__main__":
    main()

