import torch
import numpy as np
import pandas as pd
from pathlib import Path
from anomalib.data import Folder
from anomalib.models import EfficientAd
from anomalib.engine import Engine
from torchvision import transforms

# --- CONFIGURATION ---
PROJECT_ROOT = Path("E:/Project-Work")
DATASET_ROOT = PROJECT_ROOT / "Final_Dataset" / "inspector_by_connector"
RESULTS_ROOT = PROJECT_ROOT / "results"

data_transforms = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((256, 256)),
    transforms.ToTensor(), 
])

connectors = [d.name for d in DATASET_ROOT.iterdir() if d.is_dir()]
results_summary = []

print(f"Testing {len(connectors)} connectors with RISK-AVERSE logic...")

for conn_name in connectors:
    checkpoint_path = RESULTS_ROOT / conn_name / "weights" / "best_model.ckpt"
    if not checkpoint_path.exists():
        continue

    # Setup Data
    datamodule = Folder(
        name=conn_name,
        root=str(DATASET_ROOT / conn_name),
        normal_dir="train/good",
        normal_test_dir="test/good",
        abnormal_dir="test/bad",
        augmentations=data_transforms, 
        num_workers=0,
        train_batch_size=1,
        eval_batch_size=1,
        seed=412,
        val_split_mode="same_as_test"
    )
    
    try:
        model = EfficientAd.load_from_checkpoint(str(checkpoint_path))
        model.eval()
        model.cuda()
        
        engine = Engine(accelerator="gpu", devices=1, logger=False)
        predictions = engine.predict(model=model, datamodule=datamodule)
        
        good_scores = []
        bad_scores = []
        
        for batch in predictions:
            if "pred_score" in batch.keys():
                scores = batch["pred_score"].cpu().numpy()
                labels = batch["gt_label"].cpu().numpy()
                for score, label in zip(scores, labels):
                    if label == 0: good_scores.append(score)
                    else: bad_scores.append(score)
        
        # --- NEW LOGIC: TOP-DOWN THRESHOLDING ---
        n_good = len(good_scores)
        n_bad = len(bad_scores)
        
        status = "UNKNOWN"
        safe_threshold = 0.0
        false_positive_count = 0
        
        if n_bad > 0:
            # 1. Find the "Floor" of the defects
            # We want to catch 100% of known defects.
            min_bad_score = np.min(bad_scores)
            
            # 2. Set Threshold slightly lower (Safety Margin)
            # We use 0.95 factor to be safe against slightly easier defects
            safe_threshold = min_bad_score * 0.95
            
            # 3. Check how many "Good" images fail this threshold
            # These are your "Outliers" or False Positives
            if n_good > 0:
                false_positives = np.array(good_scores) > safe_threshold
                false_positive_count = np.sum(false_positives)
                fp_rate = false_positive_count / n_good
                
                if fp_rate == 0:
                    status = "✅ PERFECT"
                elif fp_rate < 0.10:
                    status = "⚠️ WORKS (With FP)"
                else:
                    status = "❌ POOR SEPARATION"
            else:
                status = "⚠️ NO GOOD DATA"
                
        else:
            status = "⚠️ NO BAD DATA"
            # Fallback: If we have no bad data, we default to the paper's normalization assumption
            # But this is risky without validation
            if n_good > 0:
                safe_threshold = 0.5 # Default conservative guess
            else:
                safe_threshold = 0.0

        print(f"[{conn_name}] {status}")
        print(f"   Min Bad Score: {min_bad_score if n_bad > 0 else 0:.4f}")
        print(f"   Set Threshold: {safe_threshold:.4f}")
        print(f"   False Positives: {false_positive_count}/{n_good}")
        
        results_summary.append({
            "Connector": conn_name,
            "Status": status,
            "Threshold": safe_threshold,
            "Min_Bad_Score": min_bad_score if n_bad > 0 else 0,
            "False_Positives": false_positive_count,
            "Total_Good": n_good,
            "Total_Bad": n_bad
        })
        
        del model, engine, predictions
        torch.cuda.empty_cache()
        
    except Exception as e:
        print(f"Error {conn_name}: {e}")

# --- SAVE ---
df = pd.DataFrame(results_summary)
df.sort_values(by="False_Positives", ascending=True, inplace=True) 
print("\n")
print(df.to_string(index=False))
df.to_csv("robust_thresholds.csv", index=False)

import torch
import numpy as np
import pandas as pd
import cv2
import shutil
from pathlib import Path
from anomalib.data import Folder
from anomalib.models import EfficientAd
from anomalib.engine import Engine
from torchvision import transforms

# --- CONFIGURATION ---
PROJECT_ROOT = Path("E:/Project-Work")
DATASET_ROOT = PROJECT_ROOT / "Final_Dataset" / "inspector_by_connector"
RESULTS_ROOT = PROJECT_ROOT / "results"
OUTPUT_DIR = PROJECT_ROOT / "final_visualizations"
THRESHOLDS_FILE = "robust_thresholds.csv"

# *** SELECTIVE FILTER ***
# Only process these connectors. Comment out to process all.
TARGET_CONNECTORS = ["conn1","conn2", "conn3", "conn4", "conn5", "conn6", "conn7", "conn8", "conn9"] 

# 1. Verify Thresholds
if not Path(THRESHOLDS_FILE).exists():
    raise FileNotFoundError("robust_thresholds.csv not found!")

df_thresh = pd.read_csv(THRESHOLDS_FILE)

# 2. Reset Output Directory (Only for targets to avoid wiping others if you want)
if not OUTPUT_DIR.exists():
    OUTPUT_DIR.mkdir(parents=True)

data_transforms = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((256, 256)),
    transforms.ToTensor(), 
])

# --- GENERATION LOOP ---
for index, row in df_thresh.iterrows():
    conn_name = row['Connector']
    
    # --- FILTER LOGIC ---
    if conn_name not in TARGET_CONNECTORS:
        continue
    # --------------------

    threshold = float(row['Threshold'])
    print(f"\n==========================================")
    print(f"[{conn_name}] Generating visuals... (Thresh: {threshold:.4f})")
    
    if threshold <= 0.0:
        print("   -> Skipping (Invalid Threshold or No Data)")
        continue

    # Setup Data
    datamodule = Folder(
        name=conn_name,
        root=str(DATASET_ROOT / conn_name),
        normal_dir="train/good",
        normal_test_dir="test/good",
        abnormal_dir="test/bad",
        augmentations=data_transforms, 
        num_workers=0,
        train_batch_size=1,
        eval_batch_size=1,
        seed=412,
        val_split_mode="same_as_test"
    )
    
    # Load Model
    checkpoint_path = RESULTS_ROOT / conn_name / "weights" / "best_model.ckpt"
    if not checkpoint_path.exists():
        print(f"   ❌ No model found at {checkpoint_path}")
        continue

    # Load Weights
    print("   -> Loading Model...")
    model = EfficientAd.load_from_checkpoint(str(checkpoint_path))
    model.eval()
    model.cuda()
    
    # Predict
    engine = Engine(accelerator="gpu", devices=1, logger=False)
    predictions = engine.predict(model=model, datamodule=datamodule)
    
    # Setup Output Folders
    save_root = OUTPUT_DIR / conn_name
    # Clean old visuals for this specific connector only
    if save_root.exists(): shutil.rmtree(save_root)
    (save_root / "good").mkdir(parents=True, exist_ok=True)
    (save_root / "bad").mkdir(parents=True, exist_ok=True)
    
    count = 0
    print("   -> Processing Batches...")
    
    for batch_idx, batch in enumerate(predictions):
        try:
            # DIRECT ACCESS (Fixes the "in batch" bug)
            imgs = batch["image"]
            anomaly_maps = batch["anomaly_map"]
            gt_labels = batch["gt_label"]
            pred_scores = batch["pred_score"] # Singular key
            
            # Handle Path variations
            if "image_path" in batch.keys(): 
                paths = batch["image_path"]
            elif "path" in batch.keys(): 
                paths = batch["path"]
            else: 
                paths = [f"img_{batch_idx}_{i}.png" for i in range(len(imgs))]

            for i in range(len(imgs)):
                img = imgs[i].cpu().numpy().transpose(1, 2, 0) * 255
                img = img.astype(np.uint8)
                img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                
                ano_map = anomaly_maps[i].cpu().numpy()
                score = pred_scores[i].item()
                label = gt_labels[i].item()
                
                path_obj = Path(paths[i]) if isinstance(paths[i], (str, Path)) else Path(str(paths[i]))
                filename = path_obj.name

                # --- VISUALIZATION ---
                mask = (ano_map > threshold).astype(np.uint8) * 255
                
                # Heatmap
                norm_map = (ano_map / (threshold * 2.0)).clip(0, 1)
                heatmap = cv2.applyColorMap((norm_map * 255).astype(np.uint8), cv2.COLORMAP_JET)
                overlay = cv2.addWeighted(img, 0.6, heatmap, 0.4, 0)
                
                contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                cv2.drawContours(overlay, contours, -1, (0, 0, 255), 2)

                color = (0, 255, 0) if score < threshold else (0, 0, 255)
                label_text = "PASS" if score < threshold else "FAIL"
                cv2.putText(overlay, f"{label_text} ({score:.2f})", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                
                subfolder = "good" if label == 0 else "bad"
                cv2.imwrite(str(save_root / subfolder / f"vis_{filename}"), overlay)
                cv2.imwrite(str(save_root / subfolder / f"mask_{filename}"), mask)
                count += 1
                
        except KeyError as e:
            print(f"   ❌ KEY ERROR: {e}")
            break
        except Exception as e:
            print(f"   ❌ ERROR: {e}")
            break
    
    print(f"   -> Saved {count} images for {conn_name}")
    
    del model, engine, predictions
    torch.cuda.empty_cache()

print(f"\nDONE! Checked: {TARGET_CONNECTORS}")