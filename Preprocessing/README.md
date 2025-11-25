# Complete Guide: Preprocessing and Feature Engineering Pipeline for PCB Connector Recognition

This guide describes the complete process from downloading the raw images to feature analysis and model training for distinguishing OK/KO connectors.

---

## 📋 Table of Contents

1. [Prerequisites](#prerequisites)
2. [Project Structure](#project-structure)
3. [Step 0: Separate TOP and BACK Images](#step-0-separate-top-and-back-images)
4. [Step 1: Image Alignment](#step-1-image-alignment)
5. [Step 2: Connector Extraction](#step-2-connector-extraction)
6. [Step 3: Scalar Feature Extraction](#step-3-scalar-feature-extraction)
7. [Step 4: Manual Labeling](#step-4-manual-labeling)
8. [Step 5: Feature Engineering and Models](#step-5-feature-engineering-and-models)
9. [Step 6: Feature Selection and SHAP](#step-6-feature-selection-and-shap)
10. [Troubleshooting](#troubleshooting)

---

## Prerequisites

### Python Dependencies

Make sure you have Python 3.10+ installed along with the following libraries:

```bash
pip install numpy opencv-python pandas matplotlib tqdm scikit-learn shap
```

Or create a `requirements.txt` file:

```txt
numpy>=1.21.0
opencv-python>=4.5.0
pandas>=1.3.0
matplotlib>=3.4.0
tqdm>=4.62.0
scikit-learn>=1.0.0
shap>=0.40.0
```

And install with:

```bash
pip install -r requirements.txt
```

### Initial Directory Structure

Make sure you have the following structure:

```
Project Work/
├── Codice/
│   ├── preprocess_alignment.py
│   ├── crop_connectors.py
│   ├── labeler_grid.py
│   └── roi_config.json
├── Data/
│   └── TOP 1/              # ← Initial RAW images go here
│       ├── 20251106110559_TOP.png
│       ├── 20251106110633_TOP.png
│       ├── ...
│       └── Separazione_TopBack.py       
├── feature_search.py
└── feature_select_full.py
```

---

## Step 0: Separate TOP and BACK Images

**Objective**: Download the raw image folder and separate TOP images from BACK images.

### 0.1 Download and Preparation

1. Download or copy the folder `Product 1_Blu (1)` containing the raw PCB images
2. Place this folder in a convenient location (e.g., `Data/Product 1_Blu (1)/`)

### 0.2 Execution

Navigate to the `Product 1_Blu (1)` folder and run the separation script:

```bash
cd "Data/Product 1_Blu (1)"
python3 ../Separazione_TopBack.py
```

**Note**: The script `Separazione_TopBack.py` should be in the `Data/` folder. If it's in a different location, adjust the path accordingly.

**What the script does**:
- Scans all PNG files in the current directory
- Moves files ending with `_TOP.PNG` to a `TOP/` folder
- Moves files ending with `_BACK.PNG` to a `BACK/` folder

### 0.3 Verify Output

After execution, you should have:

```
Product 1_Blu (1)/
├── TOP/
│   ├── 20251106110559_TOP.png
│   ├── 20251106110633_TOP.png
│   └── ...
└── BACK/
    ├── 20251106110559_BACK.png
    ├── 20251106110633_BACK.png
    └── ...
```

### 0.4 Copy TOP Images to Project

Copy all images from the `TOP/` folder to `Data/TOP 1/`:

```bash
# From the project root
cp "Data/Non roba nostra/Product 1_Blu (1)/TOP/"*.png "Data/TOP 1/"
```

Or manually copy the contents of `TOP/` to `Data/TOP 1/`.

### 0.5 Optional: Remove BACK Folder

Since we only need TOP images for this project, you can optionally delete the `BACK/` folder to save space:

```bash
# From Product 1_Blu (1) folder
rm -rf BACK/
```

**Note**: Make sure you have copied the TOP images to `Data/TOP 1/` before deleting anything.

---

## Step 1: Image Alignment

**Objective**: Align all RAW images to the same position and normalize illumination.

### 1.1 Preparation

1. Make sure you have completed Step 0 and copied all TOP images to `Data/TOP 1/`
2. Choose a reference image (e.g., `20251106131917_TOP.png`)

### 1.2 Execution

Run the alignment script:

```bash
python3 Codice/preprocess_alignment.py \
  --input-dir "Data/TOP 1" \
  --output-dir "Data/aligned_top" \
  --reference "Data/TOP 1/20251106131917_TOP.png" \
  --crop "302,288,1883,942"
```

**Parameters**:
- `--input-dir`: Folder with RAW images (e.g., `Data/TOP 1`)
- `--output-dir`: Output folder for aligned images (e.g., `Data/aligned_top`)
- `--reference`: Path to the reference image (must be one of the images in `--input-dir`)
- `--crop`: (Optional) Board ROI in the format `xmin,ymin,xmax,ymax`. If omitted, keeps the entire image.

### 1.3 Verify Output

After execution, you should have:
- `Data/aligned_top/` with all aligned images (same original filename)
- `Data/aligned_top/alignment_results.json` with alignment results

Check some aligned images to verify they are all in the same position.

---

## Step 2: Connector Extraction

**Objective**: Extract the 9 connector regions from each aligned image.

### 2.1 ROI Configuration

The file `Codice/roi_config.json` contains the relative coordinates (0-1) of the 9 connector regions. This file is already configured by me and should not be modified unless the connector layout on the board changes.

### 2.2 Execution

Run the extraction script:

```bash
python3 Codice/crop_connectors.py \
  --input-dir "Data/aligned_top" \
  --output-dir "Data/connectors" \
  --roi-config "Codice/roi_config.json" \
  --margin 8
```

**Parameters**:
- `--input-dir`: Folder with aligned images (e.g., `Data/aligned_top`)
- `--output-dir`: Output folder for connector crops (e.g., `Data/connectors`)
- `--roi-config`: Path to the JSON file with ROI coordinates
- `--margin`: (Optional, default=0) Margin in pixels to add around each ROI

### 2.3 Verify Output

After execution, you should have:

```
Data/connectors/
├── conn1/
│   ├── 20251106110559_TOP.png
│   ├── 20251106110633_TOP.png
│   └── ...
├── conn2/
│   └── ...
├── ...
└── conn9/
    └── ...
```

Each `connX/` folder contains the crops of that connector from all aligned images.

---

## Step 3: Scalar Feature Extraction

**Objective**: Calculate simple numerical features (mean, standard deviation, etc.) for each ROI.

### 3.1 Script Creation

Create the file `Codice/extract_scalar_features.py` (if it doesn't exist) following the specifications in `Istruzioni distinzione semplice.md`.

### 3.2 Execution

Run the feature extraction:

```bash
python3 Codice/extract_scalar_features.py \
  --input_dir "Data/connectors" \
  --output_csv "features.csv" \
  --band_rel_ymin 0.35 \
  --band_rel_ymax 0.55
```

**Parameters**:
- `--input_dir`: Root folder containing subfolders `conn1/`, `conn2/`, etc.
- `--output_csv`: Output CSV file path (e.g., `features.csv`)
- `--band_rel_ymin` and `--band_rel_ymax`: (Optional) Relative vertical coordinates [0,1] of a horizontal band where the cable usually passes. If omitted, band features are not calculated.

### 3.3 Verify Output

The `features.csv` file should contain columns:
- `sample_id`: Sample ID (e.g., `20251106110559_TOP`)
- `connector_name`: Connector name (`conn1`, `conn2`, ..., `conn9`)
- `filename`: Image filename
- `gray_mean`: Mean grayscale intensity [0,1]
- `gray_std`: Standard deviation of intensity
- `band_mean`: (If specified) Mean in the horizontal band
- `band_std`: (If specified) Standard deviation in the band

**Note**: The `label` column will be empty at this point. It will be populated in the next step.

---

## Step 4: Manual Labeling

**Objective**: Manually assign OK/KO/OCCLUSION labels to each connector.

**Note**: A ready-made `features_labeled.csv` file with labels is already provided in the project. This step is **optional** and mainly for testing the labeling tool or making modifications. You can skip directly to Step 5 if you want to use the existing labeled dataset.

### 4.1 Preparation

Make sure the `features.csv` file exists in the project root directory. If you want to use the existing labels, you can skip this step and proceed to Step 5.

### 4.2 Running the Labeler

**Optional**: Run the interactive labeler to test it or modify existing labels:

```bash
python3 Codice/labeler_grid.py
```

**Note**: If you're using the provided `features_labeled.csv`, you don't need to run this step unless you want to review or modify the labels.

### 4.3 How to Use the Labeler

The labeler will open a GUI window with a 3x3 grid showing the 9 connectors of each board.

**Controls**:
- **O** = Assign label **OK** (connector properly attached)
- **K** = Assign label **KO** (connector detached or problem)
- **C** = Assign label **OCCLUSION** (cable occluding the view - to be analyzed separately)
- **B** = Go back to the previous cell (to correct errors)
- **Esc** = Save progress and exit

**Workflow**:
1. For each board, label all 9 connectors
2. The system automatically advances to the next cell after each label
3. When all 9 connectors are labeled, it automatically moves to the next board
4. You can press **B** to go back and modify a previous label

**Visualization**:
- **Green** border = OK
- **Red** border = KO
- **Orange** border = OCCLUSION
- **Yellow** border = Currently selected cell
- **Gray** border = Not yet labeled

### 4.4 Verify Output

After labeling, the file `features_labeled.csv` will be created/updated in the project root.

**Important**: 
- OCCLUSION labels are saved but **not used** for model training (only OK/KO are used)
- You can interrupt labeling at any time by pressing Esc - progress is automatically saved
- You can reopen the labeler to continue or modify existing labels

---

## Step 5: Feature Engineering and Models

**Objective**: Create derived features and train simple models to evaluate feature importance.

### 5.1 Execution

Run the feature engineering script:

```bash
python3 feature_search.py
```

### 5.2 What the Script Does

1. **Loads** `features_labeled.csv`
2. **Creates derived features**:
   - `band_minus_gray = band_mean - gray_mean`
   - `band_div_gray = band_mean / (gray_mean + 1e-6)`
   - `std_ratio = band_std / (gray_std + 1e-6)`
   - `inv_std = 1.0 / (gray_std + 1e-6)`
   - `contrast_band = band_std / (band_mean + 1e-6)`
   - `local_contrast = gray_std / (gray_mean + 1e-6)`
3. **Train/validation split** based on `board_id` (80/20)
4. **Trains models**:
   - LogisticRegression (L1) with balanced class weighting
   - RandomForest (300 trees, depth 5) with balanced class weighting
5. **Calculates importances**:
   - Logistic Regression coefficients
   - Random Forest feature importances
   - Permutation importance

### 5.3 Generated Output

The script creates the `results/` folder with:

```
results/
├── feature_engineered.csv          # Dataset with all features (base + derived)
├── feature_coefficients_logreg.csv  # Logistic Regression coefficients
├── feature_importance_rf.csv       # Random Forest importances
├── feature_importance_perm.csv     # Permutation importance
└── plots/
    └── feature_importance_rf.png   # Top 15 features plot
```

### 5.4 Interpreting Results

**Printed metrics**:
- **Acc**: Accuracy
- **F1**: F1-score (harmonic mean of precision and recall)
- **Prec**: Precision
- **Rec**: Recall
- **ROC-AUC**: Area under the ROC curve

**Confusion Matrix**:
```
[[True Neg,  False Pos],
 [False Neg, True Pos]]
```

**Note**: If you see F1=0.000, you probably have too few KO examples. See [Troubleshooting](#troubleshooting) section.

---

## Step 6: Feature Selection and SHAP

**Objective**: Automatically select the best features and analyze interpretability with SHAP.

### 6.1 Prerequisites

Make sure you have executed Step 5, which generates `results/feature_engineered.csv`.

### 6.2 Execution

Run the feature selection script:

```bash
python3 feature_select_full.py
```

### 6.3 What the Script Does

1. **Loads** `results/feature_engineered.csv`
2. **Runs RFECV** (Recursive Feature Elimination with Cross-Validation):
   - Uses RandomForest as base
   - 5-fold cross-validation
   - Scoring: F1
   - Automatically selects optimal number of features
3. **Trains final model** only on selected features
4. **Calculates SHAP values** for interpretability:
   - Summary plot
   - Bar plot
   - Ranking CSV

### 6.4 Generated Output

The script updates the `results/` folder with:

```
results/
├── selected_features.txt          # List of selected features (one per line)
├── feature_rank_shap.csv          # Feature ranking by SHAP importance
└── plots/
    ├── rfecv_curve.png            # RFECV curve (score vs number of features)
    ├── shap_summary.png           # SHAP summary plot
    └── shap_bar.png               # SHAP bar plot
```

### 6.5 Interpreting Results

**RFECV Curve**: Shows how the score (F1) changes as the number of features varies. The optimal point is where the score is maximum.

**SHAP Plots**: Show which features contribute most to model predictions.

**selected_features.txt**: Contains the list of recommended features for future models.

---

## Troubleshooting

### Problem: F1 Score = 0.000

**Cause**: Extremely imbalanced dataset (too many OK, too few KO).

**Solution**:
1. Add more KO examples using the labeler
2. The script already uses `class_weight="balanced"` to mitigate the problem
3. Check the Confusion Matrix to see if the model is at least predicting some KO

**Check distribution**:
```bash
python3 -c "import pandas as pd; df = pd.read_csv('features_labeled.csv'); print(df['label'].value_counts())"
```

### Problem: Permutation Importance = 0.0000 for all features

**Cause**: With too few KO examples in the validation set, permutation importance cannot be calculated correctly.

**Solution**: Add more KO examples (at least 20-30 in the validation set, so at least 100-150 total).

### Problem: "No labeled rows with label in {OK, KO} found"

**Cause**: The `features_labeled.csv` file has no OK or KO labels (all empty or only OCCLUSION).

**Solution**: Run the labeler (`Codice/labeler_grid.py`) to assign labels.

### Problem: "features.csv not found"

**Cause**: The `features.csv` file does not exist in the project root.

**Solution**: 
1. Run Step 3 to generate `features.csv`
2. Or copy `features_labeled.csv` to `features.csv` if you have already done labeling:
   ```bash
   cp features_labeled.csv features.csv
   ```

### Problem: Images not aligned correctly

**Cause**: The reference image might not be representative or there are too many variations between images.

**Solution**:
1. Try a different reference image
2. Verify that all images in `TOP 1/` are of the same board type
3. Manually check some aligned images in `aligned_top/`

### Problem: Connectors not extracted correctly

**Cause**: ROI coordinates in `roi_config.json` might not match the aligned images.

**Solution**:
1. Manually verify some crops in `Data/connectors/connX/`
2. If necessary, update `roi_config.json` with correct coordinates (relative coordinates 0-1)

---

## Final Project Structure

After completing all steps, the structure should be:

```
Project Work/
├── Codice/
│   ├── preprocess_alignment.py
│   ├── crop_connectors.py
│   ├── labeler_grid.py
│   ├── extract_scalar_features.py
│   └── roi_config.json
├── Data/
│   ├── TOP 1/                    # Initial RAW TOP images (from Step 0)
│   ├── Separazione_TopBack.py    # Script to separate TOP/BACK images
│   ├── aligned_top/               # Aligned images
│   │   └── alignment_results.json
│   └── connectors/                # Connector crops
│       ├── conn1/
│       ├── conn2/
│       ├── ...
│       └── conn9/
├── features.csv                   # Base features (without labels)
├── features_labeled.csv          # Features with OK/KO/OCCLUSION labels
├── feature_search.py             # Feature engineering + models
├── feature_select_full.py        # Feature selection + SHAP
├── results/                      # Analysis output
│   ├── feature_engineered.csv
│   ├── feature_coefficients_logreg.csv
│   ├── feature_importance_rf.csv
│   ├── feature_importance_perm.csv
│   ├── selected_features.txt
│   ├── feature_rank_shap.csv
│   └── plots/
│       ├── feature_importance_rf.png
│       ├── rfecv_curve.png
│       ├── shap_summary.png
│       └── shap_bar.png
└── README.md                     # This file
```

---

## Important Notes

1. **Class Imbalance**: With few KO examples, models may struggle. Try to have at least 50-100 KO examples for better results.

2. **OCCLUSION Label**: OCCLUSION labels are saved but not used for training. They can be analyzed separately in the future.

3. **Board-based Split**: Train/validation split is done by `board_id`, not by individual samples. This avoids data leakage.

4. **Feature Selection**: Features selected by RFECV are those recommended for future models. You can use them to reduce dimensionality.

5. **SHAP Analysis**: SHAP requires time for large datasets. If the validation set is very large, consider using a subsample.

---

## Support

For problems or questions:
1. Check the [Troubleshooting](#troubleshooting) section
2. Verify that all input files exist
3. Check error messages for specific details
4. Text me

---

**Last updated**: November 2025
