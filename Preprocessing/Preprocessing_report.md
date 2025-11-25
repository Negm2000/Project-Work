<div align="center" style="margin: 40px 0;">

<img src="https://raw.githubusercontent.com/Giovanni000/challenge_mia/main/Immagini%20Beko%20README/logo_polimi.png" alt="Politecnico di Milano" style="width: 30%; max-width: 300px; margin-bottom: 20px;"/>

# Automation and Control Engineering

## Project Work Report – Beko Europe

### Group 3 – Camera Placement and Vision Setup

**Academic Year 2025–2026**

---

**Authors:**  
Karim Negm  
Tingyu Chen  
Giovanni Passuello  
Feruza Sulaymonova

**Supervisor:** Prof. Fredy Ruiz  
**Co-advisors:** Marco Cederle, Davide Cazzaniga

---

*Politecnico di Milano – Department of Electronics, Information and Bioengineering (DEIB)*

</div>

---

# PREPROCESSING

## Introduction

This document describes the initial preprocessing analyses performed by Group 3 on the PCB connector images obtained during the project work. The focus of this report is on the image preprocessing pipeline developed to standardize and prepare the raw images for subsequent analysis.

The following sections will cover:
- Image alignment and illumination normalization techniques
- Connector extraction and ROI cropping procedures
- Feature extraction and data labeling methodologies
- Preliminary exploratory analysis of the dataset
- Theoretical considerations for future machine learning approaches

The preprocessing stage is fundamental for ensuring consistent, comparable images across the entire dataset, addressing challenges related to spatial variability (different board positions/orientations) and illumination variability (different lighting conditions between shots).

---

## Methodology

### 1. Image Preprocessing: Alignment and Illumination Normalization

The preprocessing stage is critical for ensuring consistent, comparable images across the entire dataset. This step addresses two major challenges: **spatial variability** (different board positions/orientations) and **illumination variability** (different lighting conditions between shots).

**Visual Comparison: Before and After Preprocessing**

The following images demonstrate the transformation from raw images to aligned and normalized images. The preprocessing pipeline ensures consistent board positioning and illumination across all images:

**Raw Images (Before Preprocessing)**:
<div style="display: flex; justify-content: space-around; align-items: center; flex-wrap: wrap;">
  <img src="https://raw.githubusercontent.com/Giovanni000/challenge_mia/main/Immagini%20Beko%20README/20251106110559_TOP.png" alt="Raw Image 1" style="width: 45%; margin: 5px;"/>
  <img src="https://raw.githubusercontent.com/Giovanni000/challenge_mia/main/Immagini%20Beko%20README/20251106112109_TOP.png" alt="Raw Image 2" style="width: 45%; margin: 5px;"/>
</div>

**Aligned Images (After Preprocessing)**:
<div style="display: flex; justify-content: space-around; align-items: center; flex-wrap: wrap;">
  <img src="https://raw.githubusercontent.com/Giovanni000/challenge_mia/main/Immagini%20Beko%20README/20251106110900_TOP.png" alt="Aligned Image 1" style="width: 45%; margin: 5px;"/>
  <img src="https://raw.githubusercontent.com/Giovanni000/challenge_mia/main/Immagini%20Beko%20README/20251106112712_TOP.png" alt="Aligned Image 2" style="width: 45%; margin: 5px;"/>
</div>

*Top row: Raw images with varying board positions and lighting conditions. Bottom row: Aligned and normalized images with consistent positioning and illumination.*

#### 1.1 Image Alignment (Raddrizzamento)

**Problem**: Raw PCB images are captured with varying board positions, orientations, and camera angles. Without alignment, the same connector would appear at different pixel locations across images, making automated analysis impossible.

**Solution**: We use **feature-based homography estimation** to align every image to a canonical reference pose. The process involves:

1. **Feature Detection**: ORB (Oriented FAST and Rotated BRIEF) features are detected in both the reference image and each candidate image:
   ```python
   orb = cv2.ORB_create(nfeatures=4000, fastThreshold=5, scaleFactor=1.2)
   kp1, des1 = orb.detectAndCompute(template_gray, None)
   kp2, des2 = orb.detectAndCompute(candidate_gray, None)
   ```

2. **Feature Matching**: Features are matched using a brute-force matcher with Hamming distance. We use a two-stage approach:
   - First, try cross-check matching (more strict)
   - If insufficient matches (<15), fall back to ratio test matching (Lowe's ratio test with threshold 0.75)
   ```python
   matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
   matches = matcher.match(des1, des2)
   # Filter top 15% of matches by distance
   matches = sorted(matches, key=lambda m: m.distance)
   keep = max(4, int(len(matches) * 0.15))
   ```

3. **Homography Estimation**: A homography matrix (3×3 transformation) is computed using RANSAC to handle outliers:
   ```python
   H, mask = cv2.findHomography(dst_pts, src_pts, cv2.RANSAC, 4.0)
   ```

4. **Image Warping**: The candidate image is warped to match the reference pose:
   ```python
   warped = cv2.warpPerspective(image, H, (template.shape[1], template.shape[0]))
   ```

5. **Fallback Method**: If feature matching fails, we use **Enhanced Correlation Coefficient (ECC)** optimization for dense alignment:
   ```python
   warp_matrix = np.eye(3, dtype=np.float32)
   _, warp_matrix = cv2.findTransformECC(
       template, candidate, warp_matrix, 
       cv2.MOTION_HOMOGRAPHY, criteria, None, 5
   )
   ```

**Why This Matters**: Alignment ensures that every connector appears at the same pixel location across all images, enabling automated ROI extraction and consistent feature computation.

**Alignment Process in Action**:

The following screenshot shows the alignment script processing images, demonstrating the automated feature matching and homography estimation:

<img src="https://raw.githubusercontent.com/Giovanni000/challenge_mia/main/Immagini%20Beko%20README/Screenshot%2015%20alle%2003.18.13.png" alt="Alignment Script Execution" style="width: 80%; margin: 20px auto; display: block; border: 1px solid #ddd; border-radius: 4px;"/>

*The alignment script automatically processes all images, matching features and applying homography transformations to ensure consistent board positioning.*

#### 1.2 Illumination Normalization (Normalizzazione della Luce)

**Problem**: Different lighting conditions between shots create significant intensity variations. A connector that appears bright in one image might appear dark in another due to lighting, not due to its actual state. This variability would confuse machine learning models.

**Solution**: We apply a **two-stage illumination normalization**:

1. **CLAHE (Contrast Limited Adaptive Histogram Equalization)**: Applied to the L channel in LAB color space to enhance local contrast while preventing over-amplification:
   ```python
   lab = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2LAB)
   l, a, b = cv2.split(lab)
   clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
   l = clahe.apply(l)  # Only enhance luminance
   lab = cv2.merge((l, a, b))
   balanced = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
   ```

2. **Gray-World White Balance**: Assumes that the average color in a scene should be gray, correcting color casts:
   ```python
   avg_b = np.mean(balanced[:, :, 0])
   avg_g = np.mean(balanced[:, :, 1])
   avg_r = np.mean(balanced[:, :, 2])
   avg_gray = (avg_b + avg_g + avg_r) / 3.0
   scale = np.array([avg_gray / avg_b, avg_gray / avg_g, avg_gray / avg_r])
   wb = balanced.astype(np.float32) * scale
   wb = np.clip(wb, 0, 255).astype(np.uint8)
   ```

**Why This Matters**: Normalization ensures that intensity differences between images reflect actual connector states (OK vs KO) rather than lighting variations. This is crucial for reliable feature extraction and model training.

**Result**: After preprocessing, we have a standardized dataset where:
- All boards appear in the same position and orientation
- Illumination differences are minimized
- Images are directly comparable for automated analysis

### 2. Connector Extraction: ROI Cropping and Local Normalization

After alignment, we extract individual connector regions from each aligned PCB image. This step creates focused images of each connector, ready for feature extraction.

#### 2.1 ROI Definition and Cropping

**ROI Configuration**: The 9 connector regions are defined using **relative coordinates** (0-1) stored in `roi_config.json`. This design choice allows the system to work with different image sizes without manual reconfiguration:

```json
{
  "name": "conn1",
  "x_min_rel": 0.15,
  "y_min_rel": 0.20,
  "x_max_rel": 0.25,
  "y_max_rel": 0.35
}
```

**Pixel Coordinate Conversion**: Relative coordinates are converted to pixel coordinates based on the actual image dimensions, with an optional safety margin:
```python
def to_pixel_box(self, width: int, height: int, margin: int = 0):
    x_min = int(self.x_min_rel * width) - margin
    y_min = int(self.y_min_rel * height) - margin
    x_max = int(self.x_max_rel * width) + margin
    y_max = int(self.y_max_rel * height) + margin
    return x_min, y_min, x_max, y_max
```

**Automated Cropping**: Each aligned image is processed to extract all 9 connector regions:
```python
for roi in rois:
    x_min, y_min, x_max, y_max = roi.to_pixel_box(width, height, margin=8)
    crop = image[y_min:y_max, x_min:x_max]
    normalized = normalize_roi(crop)
    cv2.imwrite(output_path, normalized)
```

**Visual Example: Extracted Connector Crops**

The following grid shows 9 cropped connector images extracted from different boards, demonstrating the consistency of the ROI extraction process:

<div style="display: grid; grid-template-columns: repeat(3, 1fr); gap: 15px; margin: 20px 0;">
  <div style="width: 100%; padding-bottom: 100%; position: relative; overflow: hidden; border: 1px solid #ddd; border-radius: 4px; background: #f5f5f5;">
    <img src="https://raw.githubusercontent.com/Giovanni000/challenge_mia/main/Immagini%20Beko%20README/20251106110633_TOP.png" alt="Connector Crop 1" style="position: absolute; top: 0; left: 0; width: 100%; height: 100%; object-fit: contain;"/>
  </div>
  <div style="width: 100%; padding-bottom: 100%; position: relative; overflow: hidden; border: 1px solid #ddd; border-radius: 4px; background: #f5f5f5;">
    <img src="https://raw.githubusercontent.com/Giovanni000/challenge_mia/main/Immagini%20Beko%20README/20251106111050_TOP.png" alt="Connector Crop 2" style="position: absolute; top: 0; left: 0; width: 100%; height: 100%; object-fit: contain;"/>
  </div>
  <div style="width: 100%; padding-bottom: 100%; position: relative; overflow: hidden; border: 1px solid #ddd; border-radius: 4px; background: #f5f5f5;">
    <img src="https://raw.githubusercontent.com/Giovanni000/challenge_mia/main/Immagini%20Beko%20README/20251106111126_TOP.png" alt="Connector Crop 3" style="position: absolute; top: 0; left: 0; width: 100%; height: 100%; object-fit: contain;"/>
  </div>
  <div style="width: 100%; padding-bottom: 100%; position: relative; overflow: hidden; border: 1px solid #ddd; border-radius: 4px; background: #f5f5f5;">
    <img src="https://raw.githubusercontent.com/Giovanni000/challenge_mia/main/Immagini%20Beko%20README/20251106111429_TOP.png" alt="Connector Crop 4" style="position: absolute; top: 0; left: 0; width: 100%; height: 100%; object-fit: contain;"/>
  </div>
  <div style="width: 100%; padding-bottom: 100%; position: relative; overflow: hidden; border: 1px solid #ddd; border-radius: 4px; background: #f5f5f5;">
    <img src="https://raw.githubusercontent.com/Giovanni000/challenge_mia/main/Immagini%20Beko%20README/pirla.png" alt="Connector Crop 5" style="position: absolute; top: 0; left: 0; width: 100%; height: 100%; object-fit: contain;"/>
  </div>
  <div style="width: 100%; padding-bottom: 100%; position: relative; overflow: hidden; border: 1px solid #ddd; border-radius: 4px; background: #f5f5f5;">
    <img src="https://raw.githubusercontent.com/Giovanni000/challenge_mia/main/Immagini%20Beko%20README/20251106111617_TOP.png" alt="Connector Crop 6" style="position: absolute; top: 0; left: 0; width: 100%; height: 100%; object-fit: contain;"/>
  </div>
  <div style="width: 100%; padding-bottom: 100%; position: relative; overflow: hidden; border: 1px solid #ddd; border-radius: 4px; background: #f5f5f5;">
    <img src="https://raw.githubusercontent.com/Giovanni000/challenge_mia/main/Immagini%20Beko%20README/20251106111729_TOP.png" alt="Connector Crop 7" style="position: absolute; top: 0; left: 0; width: 100%; height: 100%; object-fit: contain;"/>
  </div>
  <div style="width: 100%; padding-bottom: 100%; position: relative; overflow: hidden; border: 1px solid #ddd; border-radius: 4px; background: #f5f5f5;">
    <img src="https://raw.githubusercontent.com/Giovanni000/challenge_mia/main/Immagini%20Beko%20README/20251106112220_TOP.png" alt="Connector Crop 8" style="position: absolute; top: 0; left: 0; width: 100%; height: 100%; object-fit: contain;"/>
  </div>
  <div style="width: 100%; padding-bottom: 100%; position: relative; overflow: hidden; border: 1px solid #ddd; border-radius: 4px; background: #f5f5f5;">
    <img src="https://raw.githubusercontent.com/Giovanni000/challenge_mia/main/Immagini%20Beko%20README/20251106142745_TOP.png" alt="Connector Crop 9" style="position: absolute; top: 0; left: 0; width: 100%; height: 100%; object-fit: contain;"/>
  </div>
</div>


*Grid of 9 extracted connector crops showing the consistency of the ROI extraction process across different boards.*

#### 2.2 Local Illumination Normalization

**Problem**: Even after global illumination normalization, local variations within each connector crop can affect feature extraction. Small shadows, reflections, or lighting gradients within the ROI can introduce noise.

**Solution**: Each connector crop undergoes **local CLAHE normalization**:

```python
def normalize_roi(img: np.ndarray) -> np.ndarray:
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Apply CLAHE with smaller tile size for local adaptation
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    equalized = clahe.apply(gray)
    
    # Normalize to float32 [0, 1] for consistent feature computation
    normalized = equalized.astype(np.float32) / 255.0
    return normalized
```

**Why Local Normalization?**: 
- **Local Adaptation**: CLAHE adapts to local contrast within each small region (8×8 tiles), enhancing details that might be lost in global normalization
- **Consistency**: Normalizing to [0, 1] ensures that feature values (mean, std) are comparable across all connector images
- **Noise Reduction**: Local equalization reduces the impact of small shadows or reflections that might be present in specific crops

**Differences from Global Normalization**:
- **Global normalization** (alignment stage): Handles large-scale lighting differences between entire images
- **Local normalization** (crop stage): Handles fine-scale variations within each connector region

**Result**: 9 separate, normalized connector images per board, stored in organized folders (`conn1/`, `conn2/`, ..., `conn9/`), ready for feature extraction.

### 3. Feature Extraction

**Scalar Features**: For each connector image, we compute:
- **Gray-level statistics**: Mean and standard deviation of grayscale intensity
- **Band features**: Statistics in a horizontal band where cables typically pass
- **Derived features**: Ratios and differences that capture relative patterns

**Feature Engineering**: Additional features are created to improve discrimination:
- `band_minus_gray`: Difference between band and overall mean
- `band_div_gray`: Ratio of band to overall intensity
- `std_ratio`: Ratio of standard deviations
- `local_contrast`: Local contrast measure
- And more...

### 4. Data Labeling

**Interactive Labeler**: A custom GUI tool allows efficient manual labeling:
- 3x3 grid visualization of all 9 connectors per board
- Keyboard shortcuts: O (OK), K (KO), C (OCCLUSION), P (PARTIAL OCCLUSION)
- Progress tracking and automatic saving
- Ability to review and correct labels

**Labeler Interface**:

The following screenshot shows the interactive labeling tool in action, displaying all 9 connectors in a 3x3 grid with color-coded labels:

<img src="https://raw.githubusercontent.com/Giovanni000/challenge_mia/main/Immagini%20Beko%20README/Screenshot%202025-11-19%20alle%2022.39.41.png" alt="Interactive Labeler GUI" style="width: 80%; margin: 20px auto; display: block; border: 1px solid #ddd; border-radius: 4px;"/>

*The labeler provides an intuitive interface for quickly assigning labels to each connector, with visual feedback through color-coded borders and labels.*

**Label Categories**:
- **OK**: Connector properly attached
- **KO**: Connector detached or problematic
- **OCCLUSION**: Cable occluding the view (excluded from training)
- **PARTIAL OCCLUSION**: Connector partially occluded (excluded from training)

**Connector ROI Visualization**:

The following image shows all 9 connector regions highlighted on the PCB board, demonstrating the precise ROI definitions used for extraction:

<img src="https://raw.githubusercontent.com/Giovanni000/challenge_mia/main/Immagini%20Beko%20README/connectors_roi_boxes.png" alt="Connector ROI Boxes" style="width: 80%; margin: 20px auto; display: block; border: 1px solid #ddd; border-radius: 4px;"/>

*Each connector is clearly marked with a colored bounding box and label (CONN1-CONN9), showing the exact regions extracted for feature analysis.*

**Labeling Statistics by Connector**:

The following table shows the distribution of labels across all 9 connectors, providing insights into the dataset composition and highlighting which connectors are most affected by occlusions:

| Connector | OK | OK % | KO | KO % | OCCLUSION | OCCLUSION % | Total |
|-----------|----|----|----|----|-----------|-------------|-------|
| conn1 | 171 | 98.84% | 2 | 1.16% | 0 | 0.00% | 173 |
| conn2 | 170 | 98.27% | 2 | 1.16% | 1 | 0.58% | 173 |
| conn3 | 173 | 100.00% | 0 | 0.00% | 0 | 0.00% | 173 |
| conn4 | 105 | 60.69% | 0 | 0.00% | 68 | 39.31% | 173 |
| conn5 | 112 | 64.74% | 2 | 1.16% | 59 | 34.10% | 173 |
| conn6 | 167 | 96.53% | 0 | 0.00% | 6 | 3.47% | 173 |
| conn7 | 143 | 82.66% | 1 | 0.58% | 29 | 16.76% | 173 |
| conn8 | 122 | 70.52% | 1 | 0.58% | 50 | 28.90% | 173 |
| conn9 | 163 | 94.22% | 3 | 1.73% | 7 | 4.05% | 173 |

**Key Observations**:
- **conn3**: Perfect visibility (100% OK, no occlusions)
- **conn4**: Highest occlusion rate (39.31%) - frequently occluded by cables or labels
- **conn5**: Second highest occlusion rate (34.10%)
- **conn8**: Third highest occlusion rate (28.90%)
- **Overall KO rate**: Very low (0-1.73% per connector), indicating high production quality
- **Occlusion patterns**: conn4, conn5, and conn8 are most affected, likely due to their position relative to cables and labels

**Visual Examples of Occlusion Types (Connector 4)**:

The following images demonstrate the three states of connector 4: occluded by cables, occluded by labels, and free (as it should be):

<div style="display: flex; justify-content: space-around; align-items: center; flex-wrap: wrap;">
  <img src="https://raw.githubusercontent.com/Giovanni000/challenge_mia/main/Immagini%20Beko%20README/20251106135917_TOP.png" alt="Connector 4 - Occluded by Cables" style="width: 30%; margin: 5px;"/>
  <img src="https://raw.githubusercontent.com/Giovanni000/challenge_mia/main/Immagini%20Beko%20README/20251106141048_TOP.png" alt="Connector 4 - Occluded by Labels" style="width: 30%; margin: 5px;"/>
  <img src="https://raw.githubusercontent.com/Giovanni000/challenge_mia/main/Immagini%20Beko%20README/20251106141123_TOP.png" alt="Connector 4 - Free" style="width: 30%; margin: 5px;"/>
</div>

*Left to right: Connector 4 occluded by cables, occluded by labels, and free (ideal state)*

### 5. Machine Learning Pipeline

*Note: The following discussion on model training and the proposed approaches is theoretical in nature. This analysis was conducted out of curiosity on the current dataset, but it should be clearly stated that this process has not yet led to any concrete results. The models and training procedures described below represent exploratory work and future directions rather than completed implementations.*

**Models Used**:
- **Logistic Regression (L1)**: Linear model with L1 regularization for feature selection
- **Random Forest**: Ensemble method with 300 trees, depth 5

**Class Balancing**: Both models use `class_weight="balanced"` to handle the imbalanced dataset.

**Evaluation Metrics**:
- Accuracy
- F1-Score
- Precision
- Recall
- ROC-AUC
- Confusion Matrix

**Feature Importance**:
- Logistic Regression coefficients
- Random Forest feature importances
- Permutation importance (validation-based)

### 6. Feature Selection

**RFECV (Recursive Feature Elimination with Cross-Validation)**:
- Automatically selects the optimal number of features
- Uses 5-fold cross-validation
- Scoring metric: F1-score
- Reduces dimensionality while maintaining performance

### 7. Model Interpretability

**SHAP (SHapley Additive exPlanations)**:
- Explains individual predictions
- Identifies which features contribute most to each decision
- Provides global feature importance ranking
- Visualizes feature contributions

---

## Pipeline Overview

```
Raw Images (TOP/BACK)
    ↓
[Step 0] Separate TOP from BACK
    ↓
TOP Images
    ↓
[Step 1] Image Alignment & Normalization
    ↓
Aligned Images
    ↓
[Step 2] Connector Extraction (9 per board)
    ↓
Connector Crops (conn1...conn9)
    ↓
[Step 3] Feature Extraction
    ↓
features.csv (scalar features)
    ↓
[Step 4] Manual Labeling (OK/KO/OCCLUSION)
    ↓
features_labeled.csv
    ↓
[Step 5] Feature Engineering & Model Training
    ↓
feature_engineered.csv + Model Results
    ↓
[Step 6] Feature Selection & SHAP Analysis
    ↓
Selected Features + Interpretability Results
```

---

## Results and Analysis

**Model** In progress

**Visual Analysis**: The following plots show production rate and intervals, providing insights into 
production patterns:

<div style="display: flex; justify-content: space-around; align-items: center;">
  <img src="https://raw.githubusercontent.com/Giovanni000/challenge_mia/main/Immagini%20Beko%20README/production_rate.png" alt="Production Rate" style="width: 45%;"/>
  <img src="https://raw.githubusercontent.com/Giovanni000/challenge_mia/main/Immagini%20Beko%20README/production_intervals.png" alt="Production Intervals" style="width: 45%;"/>
</div>

**Summary Statistics**:

| Metric | Value |
|--------|-------|
| Total Time Span | 0 days, 5 hours |
| Total Boards | 173 |
| Average Interval (minutes) | 2.06 |
| Overall OK Ratio (%) | 99.18 |
| Peak Production Hour | 13 |



#### Comprehensive Illumination Analysis Results

A comprehensive analysis was performed comparing illumination statistics between original images (TOP 1) and processed images (Aligned), examining morning vs afternoon differences and the influence of cable presence. The results are summarized in the following table:

| Stage | Period | Mean Intensity | Std | Count | Notes |
|-------|--------|---------------|-----|-------|-------|
| **TOP 1** | Morning (6-11h) | 56.69 | 1.97 | 48 | Original images before processing |
| **TOP 1** | Afternoon (12-17h) | 54.96 | 2.41 | 129 | Original images before processing |
| **TOP 1** | **Difference** | **-1.73** | - | - | Morning is slightly brighter |
| **Aligned** | Morning (6-11h) | 82.39 | 2.13 | 48 | After global normalization |
| **Aligned** | Afternoon (12-17h) | 81.85 | 2.15 | 125 | After global normalization |
| **Aligned** | **Difference** | **-0.54** | - | - | **68% reduction in variation** |
| **Cable Influence** | Correlation (TOP 1) | 0.165 | - | - | Moderate, minimal impact |
| **Cable Influence** | Correlation (Aligned) | ~0.15 | - | - | Moderate, minimal impact |

**Key Findings and Conclusions**:

1. **Minimal Temporal Variation**: The difference between morning and afternoon illumination in original images is only **1.73 points** (on a 0-255 scale), representing approximately **3% variation**. This is a very small difference that does not pose significant challenges for image analysis.

2. **Effective Normalization**: The preprocessing pipeline (CLAHE + gray-world white balance) successfully reduces the morning/afternoon difference from **1.73 to 0.54 points** (68% reduction), demonstrating that the normalization effectively compensates for natural lighting variations.

3. **Negligible Cable Impact**: The correlation between cable presence and illumination intensity is **0.165** (moderate), with a difference of only **0.50 points** between images with high vs low cable presence. This confirms that cables do not significantly affect illumination measurements.

4. **Stable Processing**: The standard deviation remains consistent across periods (1.97-2.41 for TOP 1, 2.13-2.15 for Aligned), indicating stable and predictable illumination conditions.

**Conclusion for Blue Board Analysis**:

Based on these results, **there are no significant illumination problems or particular concerns to note** for proceeding with image analysis work on the blue board. The illumination variations are:
- **Minimal** (less than 3% variation between morning and afternoon)
- **Effectively normalized** by the preprocessing pipeline (68% reduction)
- **Unaffected by cable presence** (correlation < 0.2)
- **Stable and predictable** (low standard deviation)

The preprocessing pipeline successfully handles the natural lighting variations present in the production environment, ensuring consistent feature extraction and reliable model performance regardless of the time of day or cable configuration.

More plots are already available, but due to lack of data they are not shown here.

---

### Simple Neural Network Discovery and Limitations

During the exploratory analysis, we uncovered something surprisingly powerful: in several connectors, the distribution of the average grayscale (`gray_mean`) between OK and KO samples is so distinct that a **hyperplane naturally emerges**. In other words, for those connectors, the problem is almost "solved" in the raw data — a single feature is already enough to separate OK from KO with high confidence.

From an engineering perspective, this opens up a beautiful possibility:

**Instead of deep models or complex pipelines, a very small neural network could solve part of the task efficiently.**

#### Why This Matters

**Minimal Computational Load**: A tiny feed-forward network (or even a threshold) can make the prediction in microseconds, ideal for running at the edge with negligible energy use.

**Fast Deployment**: Simple models can be trained quickly and deployed with no need for GPUs, frameworks, or heavy dependencies.

**Proof of Signal**: The existence of such a clean hyperplane is more than just a curiosity — it validates the intuition that connector presence produces a measurable and consistent signal.

**Visual Evidence**: The scatter plots below show examples of connectors (conn1 and conn5) where the hyperplane is particularly evident. Note: there are very few KO examples to provide a rigorous evaluation, but this is an indication that, together with other evidence, suggests the possibility of employing a simple neural network here.

<div style="display: flex; justify-content: space-around; align-items: center;">
  <img src="https://raw.githubusercontent.com/Giovanni000/challenge_mia/main/Immagini%20Beko%20README/scatter_mean_std_conn1.png" alt="Connector 1 Scatter Plot" style="width: 45%;"/>
  <img src="https://raw.githubusercontent.com/Giovanni000/challenge_mia/main/Immagini%20Beko%20README/scatter_mean_std_conn5.png" alt="Connector 5 Scatter Plot" style="width: 45%;"/>
</div>

**Direct Visual Comparison (Connector 2)**:

The following images show connector 2 in its OK and KO states, demonstrating the clear visual difference that makes gray mean analysis effective for detecting the hyperplane:

<div style="display: flex; justify-content: space-around; align-items: center;">
  <img src="https://raw.githubusercontent.com/Giovanni000/challenge_mia/main/Immagini%20Beko%20README/20251106142745_TOP.png" alt="Connector 2 - OK State" style="width: 45%;"/>
  <img src="https://raw.githubusercontent.com/Giovanni000/challenge_mia/main/Immagini%20Beko%20README/20251106164620_TOP.png" alt="Connector 2 - KO State" style="width: 45%;"/>
</div>

*Left: Connector 2 in OK state (properly attached). Right: Connector 2 in KO state (detached). The difference in grayscale intensity is clearly visible, validating the hyperplane approach.*

#### The Catch: Not Every Connector Behaves This Way

This approach, however, comes with caveats:

- Some connectors do not exhibit a clear linear separability based on single features like `gray_mean`: their OK/KO distributions overlap, especially under slight shifts in lighting or cable positioning.

- The model's reliance on controlled conditions (board positioning, exposure, sensor stability) makes it sensitive to changes in production or hardware.

- This approach doesn't generalize well to more challenging tasks like partially inserted connectors, irregular orientations, or new connector types — features that real factories often demand.

#### Summary of Expected Lightweight Architecture

- **Input**: `gray_mean` or `[gray_mean, band_mean]`
- **Model**: Tiny MLP (1–2 hidden layers, 4–16 neurons)
- **Output**: OK / KO

This technique represents a valuable stepping stone — an elegant, interpretable solution suited to the connectors with clearly separable signatures.

**But it's not enough long-term, and that's where YOLO becomes not just interesting, but necessary.**

### Towards End-to-End Detection with YOLO

To overcome the limitations of manual feature engineering and address future requirements (e.g., partially inserted connectors or new defect classes), the next step is to transition toward **end-to-end deep learning approaches, with YOLO as the leading candidate**.

#### Why YOLO?

**Unified Inference**: YOLO detects and classifies connectors in a single forward pass — no need for ROI extraction, cropping, thresholding, or handcrafted features.

**Spatial Awareness**: It learns where each connector is located on the board, making it robust to slight rotations, translations, and camera misalignment.

**Scalability**: Want to detect partially inserted connectors, broken pins, or extra defects? Just expand the training dataset and retrain — no pipeline rewrite required.

**Industrial-Grade Reliability**: YOLO handles variations in brightness, backgrounds, obstruction, and perspective changes — all common in real production environments.

#### Expected Workflow

1. Annotate bounding boxes for each connector in full PCB images.
2. Format dataset in YOLO structure (`images/`, `labels/`, `data.yaml`).
3. Train a YOLOv8n model on the full image.
4. Perform detection + classification in one step during inference.

#### Benefits Over the Current Pipeline

| Current Approach | YOLO Approach |
|-----------------|---------------|
| Requires alignment + cropping + feature logic | End-to-end: detection + class in 1 step |
| Custom logic per connector and board layout | Learns connector position automatically |
| Needs manual engineering for each edge case | Learns patterns directly from pixel data |
| Limited adaptability to new defects | Expand or retrain to scale up |

**In short:**

The simple neural network-based approach proves that **the signal is there**.

But YOLO ensures that the solution remains **robust, scalable, and future-proof** — whether tomorrow's problem is a rotated board, a partially seated pin, or a new connector type nobody has seen yet.

---

## Project Structure

```
Project Work/
├── Codice/                    # Core scripts
│   ├── preprocess_alignment.py    # Image alignment
│   ├── crop_connectors.py         # Connector extraction
│   ├── labeler_grid.py            # Interactive labeler
│   ├── temporal_analysis.py       # Temporal analysis
│   └── roi_config.json            # Connector coordinates
├── Data/                      # Datasets
│   ├── TOP 1/                    # Raw TOP images
│   ├── aligned_top/               # Aligned images
│   └── connectors/                # Extracted connectors
├── feature_search.py          # Feature engineering & models
├── feature_select_full.py     # Feature selection & SHAP
├── features_labeled.csv        # Labeled dataset
└── results/                   # Analysis results
    ├── feature_engineered.csv
    ├── feature_importance_*.csv
    ├── selected_features.txt
    ├── plots/                   # Feature analysis visualizations
    └── temporal_analysis/       # Temporal analysis plots
```

---

