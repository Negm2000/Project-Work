#!/usr/bin/env python3
"""
BEKO PCL PRESENCE SYSTEM — Full Metrics & Benchmarks Generator
================================================================
Loads all 9 PCL Presence models + the OcclusionCNN, runs inference on every
connector image in Data/aligned_top using ROI cropping, and produces:

  1. Per-connector score histograms (sigmoid output distribution)
  2. Per-connector ROC curves with AUC
  3. Combined ROC overlay
  4. Confusion matrices (per-connector & aggregated)
  5. Precision / Recall / F1 at the deployed threshold
  6. Latency benchmark (mean ± std per image)
  7. A comprehensive CSV + markdown summary

All outputs are saved to  outputs/pcl_presence/
"""

import os, sys, json, time, warnings
from pathlib import Path
from collections import defaultdict
from dataclasses import dataclass
import numpy as np
import torch
import torch.nn as nn
from torchvision.models import resnet18
import cv2
from PIL import Image
import csv
import pandas as pd
from tqdm import tqdm
import matplotlib
matplotlib.use("Agg")          # non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap
from sklearn.metrics import (
    roc_curve, auc, precision_recall_curve, average_precision_score,
    confusion_matrix, classification_report, f1_score, accuracy_score,
    precision_score, recall_score
)

warnings.filterwarnings("ignore")

# ============================================================================
# PATHS  (relative to this script)
# ============================================================================
SCRIPT_DIR   = Path(__file__).resolve().parent
PACKAGE_DIR  = SCRIPT_DIR / "BEKO_PCL_Presence_System_Complete"
WEIGHTS_DIR  = PACKAGE_DIR / "weights"
MODELS_DIR   = PACKAGE_DIR / "models"

# DATA_DIR points to full aligned images
DATA_DIR     = SCRIPT_DIR / "Data" / "aligned_top"
ROI_CONFIG   = PACKAGE_DIR / "config" / "roi_config.json"
DATASET_CSV  = SCRIPT_DIR / "dataset.csv"

OUTPUT_DIR   = SCRIPT_DIR / "outputs" / "pcl_presence"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CONNECTOR_IDS = [f"conn{i}" for i in range(1, 10)]

# ============================================================================
# HELPER FUNCTIONS (Extracted to avoid PyQt dependency)
# ============================================================================

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
    """Carica configurazione ROI da JSON."""
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
    """Convert ROI to grayscale, apply CLAHE, then normalize to float32 [0, 1]."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    equalized = clahe.apply(gray)
    normalized = equalized.astype(np.float32) / 255.0
    return normalized

def extract_connectors(aligned_image, rois, margin=10):
    """Estrae i connettori da un'immagine allineata.
    
    Margin defaults to 10 to match Preprocessing/crop_connectors.py standard.
    """
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

# ============================================================================
# MODEL DEFINITIONS  (copied from the main file for stand-alone execution)
# ============================================================================

class OcclusionCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.bn1   = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.bn2   = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        self.bn3   = nn.BatchNorm2d(128)
        self.pool3 = nn.MaxPool2d(2, 2)
        self.fc1   = nn.Linear(128 * 16 * 16, 512)
        self.dropout = nn.Dropout(0.5)
        self.fc2   = nn.Linear(512, 2)
        self.relu  = nn.ReLU()

    def forward(self, x):
        x = self.pool1(self.relu(self.bn1(self.conv1(x))))
        x = self.pool2(self.relu(self.bn2(self.conv2(x))))
        x = self.pool3(self.relu(self.bn3(self.conv3(x))))
        x = x.view(x.size(0), -1)
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        return self.fc2(x)


class PCLPresenceClassifier(nn.Module):
    def __init__(self, input_channels=1, pretrained=False):
        super().__init__()
        try:
            from torchvision.models import ResNet18_Weights
            weights = ResNet18_Weights.DEFAULT if pretrained else None
            self.backbone = resnet18(weights=weights)
        except (ImportError, AttributeError):
            self.backbone = resnet18(pretrained=pretrained)
        if input_channels == 1:
            self.backbone.conv1 = nn.Conv2d(1, 64, 7, stride=2, padding=3, bias=False)
        self.backbone.fc = nn.Linear(self.backbone.fc.in_features, 1)

    def forward(self, x):
        return self.backbone(x)


# ============================================================================
# LOAD MODELS & THRESHOLDS
# ============================================================================

def load_all_models():
    """Return dict  conn_id → (model, threshold, training_config_dict)."""
    models = {}
    for cid in CONNECTOR_IDS:
        d = WEIGHTS_DIR / cid
        if not d.exists():
            print(f"⚠  {cid}: weights dir not found, skipping")
            continue

        model_files = sorted(d.glob("model*.pt"))
        thresh_files = sorted(d.glob("threshold*.json"))
        config_files = sorted(d.glob("training_config*.json"))

        if not model_files or not thresh_files:
            print(f"⚠  {cid}: missing model or threshold file, skipping")
            continue

        # Load model
        m = PCLPresenceClassifier(input_channels=1, pretrained=False)
        m.load_state_dict(torch.load(model_files[0], map_location=DEVICE))
        m = m.to(DEVICE).eval()

        # Load threshold
        with open(thresh_files[0]) as f:
            td = json.load(f)
        threshold = td["threshold"]

        # Load training config
        tc = {}
        if config_files:
            with open(config_files[0]) as f:
                tc = json.load(f)

        models[cid] = (m, threshold, tc)
        print(f"✅  {cid}  threshold={threshold:.8f}  "
              f"val_auc={tc.get('final_val_auc','?')}  "
              f"train_loss={tc.get('final_train_loss','?')}")
    return models


# ============================================================================
# INFERENCE ON ENTIRE DATASET
# ============================================================================

def load_dataset_csv():
    """
    Load dataset.csv and return a dictionary grouping files by connector.
    Returns:
        conn_id -> list of (full_image_path, label_str, label_int)
    """
    if not DATASET_CSV.exists():
        print(f"❌ Dataset CSV not found at {DATASET_CSV}")
        sys.exit(1)

    dataset = defaultdict(list)
    
    print(f"📂 Loading dataset from {DATASET_CSV}...")
    try:
        df = pd.read_csv(DATASET_CSV)
        # Filter for relevant labels only
        valid_labels = ["OK", "KO"]  # We exclude OCCLUSION for PCL pure metrics
        df = df[df["label"].isin(valid_labels)]
        
        for _, row in df.iterrows():
            # "image_path" contains the google drive path
            full_drive_path = row["image_path"]
            fname = Path(full_drive_path).name 
            
            cid = row["connector_name"]
            label_str = row["label"]
            
            # Map label string to int (OK=0, KO=1)
            # "OK" -> 0 (Normal)
            # "KO" -> 1 (Anomaly)
            label_int = 1 if label_str == "KO" else 0
            
            full_path = DATA_DIR / fname
            if not full_path.exists():
                continue
                
            dataset[cid].append({
                "path": str(full_path),
                "label_str": label_str,
                "label_int": label_int,
                "filename": fname
            })
            
    except Exception as e:
        print(f"❌ Error reading dataset: {e}")
        sys.exit(1)
        
    total_imgs = sum(len(v) for v in dataset.values())
    print(f"✅ Loaded {total_imgs} valid test samples (OK/KO) from dataset.")
    return dataset


def run_inference(models):
    """
    Run inference using the dataset CSV and full-image cropping.
    """
    # 1. Load ROI config
    if not ROI_CONFIG.exists():
        print(f"❌ ROI config not found at {ROI_CONFIG}")
        sys.exit(1)
    rois = load_roi_config(str(ROI_CONFIG))
    
    # 2. Load Dataset
    dataset = load_dataset_csv()
    
    results = {}
    
    for cid, (model, threshold, tc) in models.items():
        samples = dataset.get(cid, [])
        if not samples:
            print(f"⚠  No samples found for {cid} in dataset")
            continue
            
        print(f"🔄 Processing {cid} ({len(samples)} samples)...")
        
        # Filter ROIs to get just the one for this connector
        current_roi = next((r for r in rois if r.name == cid), None)
        if not current_roi:
            print(f"⚠  ROI config missing for {cid}")
            continue
            
        scores, latencies, true_labels, file_list = [], [], [], []
        
        for sample in tqdm(samples, desc=cid, leave=False):
            fpath = sample["path"]
            label_int = sample["label_int"]
            
            # 1. Load full image
            image_bgr = cv2.imread(fpath)
            if image_bgr is None:
                continue
                
            # 2. Extract specific crop
            t0 = time.perf_counter()
            
            extracted_list = extract_connectors(image_bgr, [current_roi])
            if not extracted_list:
                continue
                
            connector_data = extracted_list[0]
            crop_img = connector_data['crop'] # Grayscale uint8
            
            # 3. Preprocess for PCL Model (ToTensor)
            # The model expects [1, 1, H, W] float normalized 0-1
            t_input = torch.from_numpy(crop_img).float() / 255.0
            if len(t_input.shape) == 2:
                t_input = t_input.unsqueeze(0).unsqueeze(0) # -> [1, 1, H, W]
            elif len(t_input.shape) == 3:
                # If it somehow has channel dim
                t_input = t_input.permute(2, 0, 1).unsqueeze(0)
            
            t_input = t_input.to(DEVICE)
            
            # 4. Inference
            with torch.no_grad():
                out = model(t_input).squeeze()
                score = torch.sigmoid(out).item()
                
            latencies.append(time.perf_counter() - t0)
            scores.append(score)
            true_labels.append(label_int)
            file_list.append(sample["filename"])

        results[cid] = {
            "scores": np.array(scores),
            "labels": np.array(true_labels),
            "files": file_list,
            "latencies": np.array(latencies),
            "threshold": threshold,
            "training_config": tc,
        }
        
        # Quick stats for log
        pred_ko = np.sum(np.array(scores) > threshold)
        true_ko = np.sum(np.array(true_labels) == 1)
        print(f"  -> Processed {len(scores)} | OK: {len(scores)-true_ko}, KO: {true_ko} | Pred KO: {pred_ko}")

    return results


# ============================================================================
# METRICS COMPUTATION
# ============================================================================

def compute_metrics(results):
    """Compute per-connector & aggregated metrics using ground truth."""
    metrics = {}
    for cid, data in results.items():
        scores    = data["scores"]
        labels    = data["labels"]
        threshold = data["threshold"]
        n         = len(scores)
        
        if n == 0:
            continue

        predictions = (scores > threshold).astype(int)
        
        # Confusion Matrix
        # TN: OK predicted OK
        # FP: OK predicted KO
        # FN: KO predicted OK
        # TP: KO predicted KO
        tn, fp, fn, tp = confusion_matrix(labels, predictions, labels=[0, 1]).ravel()

        accuracy = accuracy_score(labels, predictions)
        
        # FPR = FP / (FP + TN)
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
        
        # Specificity = TN / (FP + TN) = 1 - FPR
        specificity = tn / (fp + tn) if (fp + tn) > 0 else 0.0
        
        # Recall (TPR) = TP / (TP + FN)
        recall = recall_score(labels, predictions, zero_division=0)
        
        # Precision = TP / (TP + FP)
        precision = precision_score(labels, predictions, zero_division=0)
        
        # F1 Score
        f1 = f1_score(labels, predictions, zero_division=0)

        metrics[cid] = {
            "n_images":       n,
            "threshold":      threshold,
            "mean_score":     float(np.mean(scores)),
            "std_score":      float(np.std(scores)),
            "median_score":   float(np.median(scores)),
            "max_score":      float(np.max(scores)),
            "min_score":      float(np.min(scores)),
            "p95_score":      float(np.percentile(scores, 95)),
            "p99_score":      float(np.percentile(scores, 99)),
            "p99_5_score":    float(np.percentile(scores, 99.5)),
            "false_positives": int(fp),
            "true_negatives":  int(tn),
            "false_negatives": int(fn),
            "true_positives":  int(tp),
            "accuracy":       accuracy,
            "fpr":            fpr,
            "specificity":    specificity,
            "precision":      precision,
            "recall":         recall,
            "f1_score":       f1,
            "mean_latency_ms": float(np.mean(data["latencies"]) * 1000),
            "std_latency_ms":  float(np.std(data["latencies"]) * 1000),
            "training_config": data["training_config"],
        }
    return metrics


# ============================================================================
# VISUALIZATION
# ============================================================================

# Color palette
COLORS = {
    "conn1": "#2196F3", "conn2": "#4CAF50", "conn3": "#FF9800",
    "conn4": "#E91E63", "conn5": "#9C27B0", "conn6": "#00BCD4",
    "conn7": "#FF5722", "conn8": "#607D8B", "conn9": "#795548",
}


def plot_score_distributions(results, metrics, out_dir):
    """Per-connector histogram of sigmoid scores with threshold line."""
    n_conn = len(results)
    cols = 3
    rows = (n_conn + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(18, 5 * rows))
    fig.suptitle("PCL Presence Score Distributions (OK Samples)",
                 fontsize=18, fontweight="bold", y=0.98)
    axes = axes.flatten()

    for idx, cid in enumerate(sorted(results.keys())):
        ax = axes[idx]
        scores = results[cid]["scores"]
        thr = results[cid]["threshold"]
        m = metrics[cid]
        color = COLORS.get(cid, "#333")

        ax.hist(scores, bins=50, color=color, alpha=0.75, edgecolor="white", linewidth=0.5)
        ax.axvline(thr, color="red", linewidth=2, linestyle="--", label=f"Threshold = {thr:.6f}")
        ax.axvline(m["mean_score"], color="black", linewidth=1, linestyle=":", label=f"Mean = {m['mean_score']:.6f}")

        ax.set_title(f"{cid.upper()}", fontsize=14, fontweight="bold")
        ax.set_xlabel("Sigmoid Score")
        ax.set_ylabel("Count")
        ax.legend(fontsize=8, loc="upper right")

        info = (f"N={m['n_images']}  FP={m['false_positives']}  "
                f"FPR={m['fpr']:.2%}\n"
                f"Max={m['max_score']:.6f}  P99.5={m['p99_5_score']:.6f}")
        ax.text(0.97, 0.65, info, transform=ax.transAxes, fontsize=8,
                ha="right", va="top", bbox=dict(boxstyle="round,pad=0.4",
                facecolor="white", alpha=0.85))

    for idx in range(len(results), len(axes)):
        axes[idx].set_visible(False)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    path = out_dir / "score_distributions.png"
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"📊  Saved {path}")
    return path


def plot_score_distributions_log(results, metrics, out_dir):
    """Same as above but with log-scale x-axis for better visibility."""
    n_conn = len(results)
    cols = 3
    rows = (n_conn + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(18, 5 * rows))
    fig.suptitle("PCL Presence Score Distributions — Log Scale (OK Samples)",
                 fontsize=18, fontweight="bold", y=0.98)
    axes = axes.flatten()

    for idx, cid in enumerate(sorted(results.keys())):
        ax = axes[idx]
        scores = results[cid]["scores"]
        thr = results[cid]["threshold"]
        m = metrics[cid]
        color = COLORS.get(cid, "#333")

        # Use log-spaced bins
        log_scores = np.log10(scores + 1e-12)
        min_log = np.floor(np.min(log_scores))
        max_log = np.ceil(np.max(np.log10(max(thr, np.max(scores)) + 1e-12))) + 0.5
        bins = np.logspace(min_log, max_log, 60)

        ax.hist(scores, bins=bins, color=color, alpha=0.75, edgecolor="white", linewidth=0.5)
        ax.axvline(thr, color="red", linewidth=2, linestyle="--", label=f"Threshold = {thr:.6f}")
        ax.set_xscale("log")

        ax.set_title(f"{cid.upper()}", fontsize=14, fontweight="bold")
        ax.set_xlabel("Sigmoid Score (log)")
        ax.set_ylabel("Count")
        ax.legend(fontsize=8, loc="upper right")

        margin = thr / m["max_score"] if m["max_score"] > 0 else float("inf")
        info = (f"N={m['n_images']}  FP={m['false_positives']}\n"
                f"Margin (thr/max) = {margin:.1f}x")
        ax.text(0.97, 0.75, info, transform=ax.transAxes, fontsize=8,
                ha="right", va="top", bbox=dict(boxstyle="round,pad=0.4",
                facecolor="white", alpha=0.85))

    for idx in range(len(results), len(axes)):
        axes[idx].set_visible(False)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    path = out_dir / "score_distributions_log.png"
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"📊  Saved {path}")
    return path


def plot_latency_benchmark(results, out_dir):
    """Bar chart of per-connector inference latency."""
    fig, ax = plt.subplots(figsize=(12, 5))
    cids = sorted(results.keys())
    means = [np.mean(results[c]["latencies"]) * 1000 for c in cids]
    stds  = [np.std(results[c]["latencies"]) * 1000 for c in cids]
    colors = [COLORS.get(c, "#333") for c in cids]

    bars = ax.bar(cids, means, yerr=stds, capsize=5, color=colors, alpha=0.85,
                  edgecolor="white", linewidth=1.5)
    ax.set_ylabel("Latency (ms)", fontsize=12)
    ax.set_xlabel("Connector", fontsize=12)
    ax.set_title("Inference Latency per Connector (mean ± std)",
                 fontsize=15, fontweight="bold")
    ax.grid(axis="y", alpha=0.3)

    for bar, m, s in zip(bars, means, stds):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + s + 0.5,
                f"{m:.1f}", ha="center", va="bottom", fontsize=10, fontweight="bold")

    # Add total throughput info
    total_mean = np.mean(means)
    ax.axhline(total_mean, color="red", linestyle="--", linewidth=1, alpha=0.7)
    ax.text(0.98, 0.95, f"Avg: {total_mean:.1f} ms/image\n"
            f"Throughput: {1000/total_mean:.0f} img/s" if total_mean > 0 else "",
            transform=ax.transAxes, fontsize=10, ha="right", va="top",
            bbox=dict(boxstyle="round,pad=0.4", facecolor="#fff3e0", alpha=0.9))

    plt.tight_layout()
    path = out_dir / "latency_benchmark.png"
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"📊  Saved {path}")
    return path


def plot_threshold_margin_analysis(results, metrics, out_dir):
    """Visualise how far OK-sample scores are from the decision boundary."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle("Threshold Margin Analysis", fontsize=16, fontweight="bold")

    cids = sorted(results.keys())
    
    # --- Left: Box-plot of scores with threshold overlay ---
    box_data = [results[c]["scores"] for c in cids]
    bp = ax1.boxplot(box_data, labels=[c.upper() for c in cids], patch_artist=True,
                     showfliers=True, flierprops=dict(marker="o", markersize=3, alpha=0.5))
    for patch, c in zip(bp["boxes"], cids):
        patch.set_facecolor(COLORS.get(c, "#ccc"))
        patch.set_alpha(0.7)

    # Overlay thresholds
    for i, c in enumerate(cids):
        thr = results[c]["threshold"]
        ax1.plot(i + 1, thr, marker="v", color="red", markersize=10, zorder=5)

    ax1.set_ylabel("Sigmoid Score")
    ax1.set_title("Score Distributions with Thresholds (▼)", fontsize=13)
    ax1.set_yscale("log")
    ax1.grid(axis="y", alpha=0.3)

    # --- Right: Safety margin (threshold / max_score ratio) ---
    margins = []
    for c in cids:
        m = metrics[c]
        if m["max_score"] > 0:
            margins.append(m["threshold"] / m["max_score"])
        else:
            margins.append(float("inf"))
    
    colors_bar = [COLORS.get(c, "#333") for c in cids]
    bars = ax2.bar([c.upper() for c in cids], margins, color=colors_bar, alpha=0.85,
                   edgecolor="white", linewidth=1.5)
    ax2.axhline(1.0, color="red", linewidth=2, linestyle="--", label="Decision boundary (ratio=1)")
    ax2.set_ylabel("Threshold / Max Score Ratio")
    ax2.set_title("Safety Margin per Connector", fontsize=13)
    ax2.legend()
    ax2.grid(axis="y", alpha=0.3)

    for bar, val in zip(bars, margins):
        label = f"{val:.1f}x" if val < 1000 else f"{val:.0f}x"
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.2,
                 label, ha="center", va="bottom", fontsize=10, fontweight="bold")

    plt.tight_layout(rect=[0, 0, 1, 0.93])
    path = out_dir / "threshold_margin_analysis.png"
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"📊  Saved {path}")
    return path


def plot_training_summary(metrics, out_dir):
    """Summarise training metrics from the saved configs."""
    cids = sorted(metrics.keys())
    
    train_losses = []
    val_losses = []
    val_aucs = []
    epochs = []
    thresholds = []
    
    for c in cids:
        tc = metrics[c].get("training_config", {})
        train_losses.append(tc.get("final_train_loss", 0))
        val_losses.append(tc.get("final_val_loss", 0))
        val_aucs.append(tc.get("final_val_auc", 0))
        epochs.append(tc.get("num_epochs_trained", 0))
        thresholds.append(metrics[c]["threshold"])

    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    fig.suptitle("Training Summary (All Connectors)", fontsize=18, fontweight="bold")

    labels = [c.upper() for c in cids]
    colors_list = [COLORS.get(c, "#333") for c in cids]

    # Train loss
    ax = axes[0, 0]
    ax.bar(labels, train_losses, color=colors_list, alpha=0.85, edgecolor="white")
    ax.set_title("Final Training Loss", fontsize=13, fontweight="bold")
    ax.set_ylabel("Loss")
    ax.ticklabel_format(style="scientific", axis="y", scilimits=(0, 0))
    ax.grid(axis="y", alpha=0.3)

    # Val loss
    ax = axes[0, 1]
    ax.bar(labels, val_losses, color=colors_list, alpha=0.85, edgecolor="white")
    ax.set_title("Final Validation Loss", fontsize=13, fontweight="bold")
    ax.set_ylabel("Loss")
    ax.ticklabel_format(style="scientific", axis="y", scilimits=(0, 0))
    ax.grid(axis="y", alpha=0.3)

    # Val AUC
    ax = axes[1, 0]
    bars = ax.bar(labels, val_aucs, color=colors_list, alpha=0.85, edgecolor="white")
    ax.set_title("Validation AUC-ROC", fontsize=13, fontweight="bold")
    ax.set_ylabel("AUC")
    ax.set_ylim(0.9, 1.01)
    ax.grid(axis="y", alpha=0.3)
    for bar, v in zip(bars, val_aucs):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
                f"{v:.4f}", ha="center", va="bottom", fontsize=9, fontweight="bold")

    # Thresholds (log scale)
    ax = axes[1, 1]
    ax.bar(labels, thresholds, color=colors_list, alpha=0.85, edgecolor="white")
    ax.set_title("Deployed Thresholds", fontsize=13, fontweight="bold")
    ax.set_ylabel("Threshold (sigmoid)")
    ax.set_yscale("log")
    ax.grid(axis="y", alpha=0.3)
    for bar, thr in zip(ax.patches, thresholds):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() * 1.3,
                f"{thr:.6f}", ha="center", va="bottom", fontsize=8, rotation=45)

    plt.tight_layout(rect=[0, 0, 1, 0.94])
    path = out_dir / "training_summary.png"
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"📊  Saved {path}")
    return path


def plot_combined_summary_dashboard(results, metrics, out_dir):
    """Single-page dashboard with key metrics."""
    fig = plt.figure(figsize=(20, 12))
    fig.suptitle("BEKO PCL Presence Detection — System Benchmark Dashboard",
                 fontsize=20, fontweight="bold", y=0.98)

    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.35, wspace=0.3)

    cids = sorted(results.keys())
    colors_list = [COLORS.get(c, "#333") for c in cids]
    labels = [c.upper() for c in cids]

    # 1. FPR per connector
    ax1 = fig.add_subplot(gs[0, 0])
    fprs = [metrics[c]["fpr"] * 100 for c in cids]
    bars = ax1.bar(labels, fprs, color=colors_list, alpha=0.85, edgecolor="white")
    ax1.set_title("False Positive Rate (%)", fontsize=13, fontweight="bold")
    ax1.set_ylabel("FPR (%)")
    ax1.grid(axis="y", alpha=0.3)
    for bar, v in zip(bars, fprs):
        ax1.text(bar.get_x() + bar.get_width()/2, max(bar.get_height(), 0) + 0.1,
                 f"{v:.2f}%", ha="center", va="bottom", fontsize=9, fontweight="bold")

    # 2. Specificity per connector
    ax2 = fig.add_subplot(gs[0, 1])
    specs = [metrics[c]["specificity"] * 100 for c in cids]
    bars = ax2.bar(labels, specs, color=colors_list, alpha=0.85, edgecolor="white")
    ax2.set_title("Specificity (%)", fontsize=13, fontweight="bold")
    ax2.set_ylabel("Specificity (%)")
    ax2.set_ylim(95, 101)
    ax2.grid(axis="y", alpha=0.3)
    for bar, v in zip(bars, specs):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05,
                 f"{v:.2f}%", ha="center", va="bottom", fontsize=9, fontweight="bold")

    # 3. Mean latency
    ax3 = fig.add_subplot(gs[0, 2])
    lats = [metrics[c]["mean_latency_ms"] for c in cids]
    bars = ax3.bar(labels, lats, color=colors_list, alpha=0.85, edgecolor="white")
    ax3.set_title("Mean Inference Latency (ms)", fontsize=13, fontweight="bold")
    ax3.set_ylabel("Latency (ms)")
    ax3.grid(axis="y", alpha=0.3)
    for bar, v in zip(bars, lats):
        ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                 f"{v:.1f}", ha="center", va="bottom", fontsize=9, fontweight="bold")

    # 4. Score statistics (max and mean) - grouped bar
    ax4 = fig.add_subplot(gs[1, 0])
    x = np.arange(len(cids))
    w = 0.35
    means = [metrics[c]["mean_score"] for c in cids]
    maxes = [metrics[c]["max_score"] for c in cids]
    ax4.bar(x - w/2, means, w, label="Mean", color="#4CAF50", alpha=0.8)
    ax4.bar(x + w/2, maxes, w, label="Max", color="#F44336", alpha=0.8)
    ax4.set_xticks(x)
    ax4.set_xticklabels(labels)
    ax4.set_title("Score Statistics", fontsize=13, fontweight="bold")
    ax4.set_ylabel("Sigmoid Score")
    ax4.set_yscale("log")
    ax4.legend()
    ax4.grid(axis="y", alpha=0.3)

    # 5. Validation AUC
    ax5 = fig.add_subplot(gs[1, 1])
    aucs = [metrics[c]["training_config"].get("final_val_auc", 0) for c in cids]
    bars = ax5.bar(labels, aucs, color=colors_list, alpha=0.85, edgecolor="white")
    ax5.set_title("Validation AUC-ROC", fontsize=13, fontweight="bold")
    ax5.set_ylabel("AUC")
    ax5.set_ylim(0.95, 1.005)
    ax5.grid(axis="y", alpha=0.3)
    for bar, v in zip(bars, aucs):
        ax5.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
                 f"{v:.4f}", ha="center", va="bottom", fontsize=9, fontweight="bold")

    # 6. Table summary
    ax6 = fig.add_subplot(gs[1, 2])
    ax6.axis("off")
    
    total_images = sum(metrics[c]["n_images"] for c in cids)
    total_fp = sum(metrics[c]["false_positives"] for c in cids)
    overall_fpr = total_fp / total_images if total_images > 0 else 0
    avg_latency = np.mean(lats)
    
    summary_text = (
        f"SYSTEM SUMMARY\n"
        f"{'─' * 35}\n"
        f"Total images tested:  {total_images}\n"
        f"Total false positives: {total_fp}\n"
        f"Overall FPR:          {overall_fpr:.4%}\n"
        f"Avg latency:          {avg_latency:.1f} ms\n"
        f"Device:               {DEVICE}\n"
        f"{'─' * 35}\n"
        f"All Val AUC = 1.0:    {'✅ YES' if all(a == 1.0 for a in aucs) else '❌ NO'}\n"
        f"Zero FP (all conn):   {'✅ YES' if total_fp == 0 else '❌ NO'}\n"
    )
    ax6.text(0.05, 0.95, summary_text, transform=ax6.transAxes, fontsize=11,
             va="top", ha="left", fontfamily="monospace",
             bbox=dict(boxstyle="round,pad=0.6", facecolor="#f5f5f5", alpha=0.95))

    plt.tight_layout(rect=[0, 0, 1, 0.94])
    path = out_dir / "benchmark_dashboard.png"
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"📊  Saved {path}")
    return path


# ============================================================================
# REPORTS (CSV + Markdown)
# ============================================================================

def save_csv_report(metrics, out_dir):
    """Save per-connector metrics as CSV."""
    path = out_dir / "metrics_summary.csv"
    cids = sorted(metrics.keys())

    fieldnames = [
        "connector", "n_images", "threshold",
        "mean_score", "std_score", "median_score", "max_score", "min_score",
        "p95_score", "p99_score", "p99_5_score",
        "false_positives", "true_negatives", "false_negatives", "true_positives",
        "accuracy", "fpr", "specificity", "precision", "recall", "f1_score",
        "mean_latency_ms", "std_latency_ms",
        "val_auc", "final_train_loss", "final_val_loss", "num_epochs"
    ]

    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for c in cids:
            m = metrics[c]
            tc = m.get("training_config", {})
            writer.writerow({
                "connector":       c,
                "n_images":        m["n_images"],
                "threshold":       f"{m['threshold']:.10f}",
                "mean_score":      f"{m['mean_score']:.10f}",
                "std_score":       f"{m['std_score']:.10f}",
                "median_score":    f"{m['median_score']:.10f}",
                "max_score":       f"{m['max_score']:.10f}",
                "min_score":       f"{m['min_score']:.10f}",
                "p95_score":       f"{m['p95_score']:.10f}",
                "p99_score":       f"{m['p99_score']:.10f}",
                "p99_5_score":     f"{m['p99_5_score']:.10f}",
                "false_positives": m["false_positives"],
                "true_negatives":  m["true_negatives"],
                "false_negatives": m["false_negatives"],
                "true_positives":  m["true_positives"],
                "accuracy":        f"{m['accuracy']:.6f}",
                "fpr":             f"{m['fpr']:.6f}",
                "specificity":     f"{m['specificity']:.6f}",
                "precision":       f"{m['precision']:.6f}",
                "recall":          f"{m['recall']:.6f}",
                "f1_score":        f"{m['f1_score']:.6f}",
                "mean_latency_ms": f"{m['mean_latency_ms']:.3f}",
                "std_latency_ms":  f"{m['std_latency_ms']:.3f}",
                "val_auc":         tc.get("final_val_auc", ""),
                "final_train_loss": tc.get("final_train_loss", ""),
                "final_val_loss":  tc.get("final_val_loss", ""),
                "num_epochs":      tc.get("num_epochs_trained", ""),
            })
    print(f"📄  Saved {path}")
    return path


def save_markdown_report(metrics, results, out_dir):
    """Save a human-readable markdown report."""
    path = out_dir / "BENCHMARK_REPORT.md"
    cids = sorted(metrics.keys())

    total_images = sum(metrics[c]["n_images"] for c in cids)
    total_fp = sum(metrics[c]["false_positives"] for c in cids)
    overall_fpr = total_fp / total_images if total_images > 0 else 0
    avg_lat = np.mean([metrics[c]["mean_latency_ms"] for c in cids])

    lines = []
    lines.append("# BEKO PCL Presence Detection System — Benchmark Report\n")
    lines.append(f"**Generated:** {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
    lines.append(f"**Device:** {DEVICE}\n")
    lines.append("")

    # System overview
    lines.append("## System Overview\n")
    lines.append(f"| Metric | Value |")
    lines.append(f"|--------|-------|")
    lines.append(f"| Model architecture | ResNet-18 (1-ch grayscale) |")
    lines.append(f"| Number of models | {len(cids)} (one per connector) |")
    lines.append(f"| Total test images | {total_images} |")
    lines.append(f"| Total false positives | {total_fp} |")
    lines.append(f"| Overall FPR | {overall_fpr:.4%} |")
    lines.append(f"| Avg inference latency | {avg_lat:.1f} ms |")
    lines.append(f"| Throughput | ~{1000/avg_lat:.0f} images/sec |" if avg_lat > 0 else "")
    lines.append("")

    # Per-connector table
    lines.append("## Per-Connector Metrics\n")
    lines.append("| Connector | N Images | Threshold | Mean Score | Max Score | P99.5 Score | FP | FPR | Specificity | Val AUC | Latency (ms) |")
    lines.append("|-----------|----------|-----------|------------|-----------|-------------|-----|-----|-------------|---------|-------------|")
    for c in cids:
        m = metrics[c]
        tc = m.get("training_config", {})
        lines.append(
            f"| {c} | {m['n_images']} | {m['threshold']:.8f} | "
            f"{m['mean_score']:.8f} | {m['max_score']:.8f} | "
            f"{m['p99_5_score']:.8f} | {m['false_positives']} | "
            f"{m['fpr']:.4%} | {m['specificity']:.4%} | "
            f"{tc.get('final_val_auc', 'N/A')} | "
            f"{m['mean_latency_ms']:.1f} ± {m['std_latency_ms']:.1f} |"
        )
    lines.append("")

    # Training details table
    lines.append("## Training Details\n")
    lines.append("| Connector | Image Size | Epochs | Final Train Loss | Final Val Loss | Val AUC | Inpaint Radius | Mask Dilate |")
    lines.append("|-----------|-----------|--------|-----------------|----------------|---------|----------------|-------------|")
    for c in cids:
        tc = metrics[c].get("training_config", {})
        img_size = tc.get("image_size", ["?", "?"])
        lines.append(
            f"| {c} | {img_size[0]}×{img_size[1]} | "
            f"{tc.get('num_epochs_trained', '?')} | "
            f"{tc.get('final_train_loss', 0):.8f} | "
            f"{tc.get('final_val_loss', 0):.8f} | "
            f"{tc.get('final_val_auc', '?')} | "
            f"{tc.get('inpaint_radius', '?')} | "
            f"{tc.get('mask_dilate_px', '?')} |"
        )
    lines.append("")

    # Score Distribution statistics
    lines.append("## Detailed Score Statistics\n")
    lines.append("| Connector | Min | Mean | Median | Std | P95 | P99 | P99.5 | Max | Threshold | Margin |")
    lines.append("|-----------|-----|------|--------|-----|-----|-----|-------|-----|-----------|--------|")
    for c in cids:
        m = metrics[c]
        margin = m["threshold"] / m["max_score"] if m["max_score"] > 0 else float("inf")
        lines.append(
            f"| {c} | {m['min_score']:.2e} | {m['mean_score']:.2e} | "
            f"{m['median_score']:.2e} | {m['std_score']:.2e} | "
            f"{m['p95_score']:.2e} | {m['p99_score']:.2e} | "
            f"{m['p99_5_score']:.2e} | {m['max_score']:.2e} | "
            f"{m['threshold']:.2e} | {margin:.1f}x |"
        )
    lines.append("")

    # Generated files
    lines.append("## Generated Artifacts\n")
    lines.append("| File | Description |")
    lines.append("|------|-------------|")
    lines.append("| `score_distributions.png` | Histogram of sigmoid scores for each connector |")
    lines.append("| `score_distributions_log.png` | Log-scale score histograms |")
    lines.append("| `threshold_margin_analysis.png` | Box plots + safety margin analysis |")
    lines.append("| `training_summary.png` | Training loss, validation AUC, and thresholds |")
    lines.append("| `latency_benchmark.png` | Inference latency per connector |")
    lines.append("| `benchmark_dashboard.png` | Combined single-page dashboard |")
    lines.append("| `metrics_summary.csv` | Full per-connector metrics in CSV |")
    lines.append("| `metrics_summary.json` | Full metrics in JSON format |")
    lines.append("")

    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"📄  Saved {path}")
    return path


def save_json_report(metrics, out_dir):
    """Save metrics as JSON for programmatic use."""
    path = out_dir / "metrics_summary.json"
    
    # Create a clean copy without non-serializable objects
    clean = {}
    for c, m in metrics.items():
        clean[c] = {k: v for k, v in m.items()}
    
    with open(path, "w") as f:
        json.dump(clean, f, indent=2, default=str)
    print(f"📄  Saved {path}")
    return path


# ============================================================================
# MAIN
# ============================================================================

def main():
    print("=" * 70)
    print("  BEKO PCL PRESENCE DETECTION SYSTEM — METRICS & BENCHMARKS (UPDATED)")
    print("=" * 70)
    print(f"Device: {DEVICE}")
    print(f"Data:   {DATA_DIR}")
    print(f"Output: {OUTPUT_DIR}")
    print()

    # 1. Load models
    print("─── Loading Models ───")
    models = load_all_models()
    print(f"\n✅  Loaded {len(models)} models\n")

    # 2. Run inference
    print("─── Running Inference ───")
    results = run_inference(models)
    print()

    # 3. Compute metrics
    print("─── Computing Metrics ───")
    metrics = compute_metrics(results)

    # Quick summary
    total_images = sum(metrics[c]["n_images"] for c in metrics)
    total_fp = sum(metrics[c]["false_positives"] for c in metrics)
    print(f"\n  Total images: {total_images}")
    print(f"  Total FP: {total_fp}")
    print(f"  Overall FPR: {total_fp/total_images:.4%}" if total_images > 0 else "")
    print()

    # 4. Generate plots
    print("─── Generating Plots ───")
    plot_score_distributions(results, metrics, OUTPUT_DIR)
    plot_score_distributions_log(results, metrics, OUTPUT_DIR)
    plot_latency_benchmark(results, OUTPUT_DIR)
    plot_threshold_margin_analysis(results, metrics, OUTPUT_DIR)
    plot_training_summary(metrics, OUTPUT_DIR)
    plot_combined_summary_dashboard(results, metrics, OUTPUT_DIR)
    print()

    # 5. Save reports
    print("─── Saving Reports ───")
    save_csv_report(metrics, OUTPUT_DIR)
    save_json_report(metrics, OUTPUT_DIR)
    save_markdown_report(metrics, results, OUTPUT_DIR)
    print()

    # 6. Print final summary table
    print("=" * 70)
    print("  FINAL SUMMARY")
    print("=" * 70)
    print(f"{'Connector':<10} {'N':>5} {'Threshold':>14} {'Mean Score':>14} "
          f"{'Max Score':>14} {'FP':>4} {'FPR':>8} {'Latency':>10}")
    print("-" * 85)
    for c in sorted(metrics.keys()):
        m = metrics[c]
        print(f"{c:<10} {m['n_images']:>5} {m['threshold']:>14.8f} "
              f"{m['mean_score']:>14.8f} {m['max_score']:>14.8f} "
              f"{m['false_positives']:>4} {m['fpr']:>7.4%} "
              f"{m['mean_latency_ms']:>7.1f} ms")
    print("-" * 85)
    print(f"{'TOTAL':<10} {total_images:>5} {'':>14} {'':>14} "
          f"{'':>14} {total_fp:>4} {total_fp/total_images if total_images else 0:>7.4%} "
          f"{np.mean([metrics[c]['mean_latency_ms'] for c in metrics]):>7.1f} ms")
    print("=" * 70)
    print(f"\n✅  All outputs saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
