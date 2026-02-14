# Package Information - BEKO PCL Presence Detection System

## 📦 Package Contents

This is a **complete, standalone package** containing:

### ✅ Core Application
- `beko_pcl_presence_system.py` - Main GUI application (self-contained)

### ✅ Models
- **Occlusion Detection**: `models/occlusion_cnn.pth`
- **PCL Presence Models**: 9 connectors (conn1 through conn9)
  - Each connector has: `model*.pt` and `threshold*.json`

### ✅ Configuration
- `config/roi_config.json` - ROI definitions for all connectors

### ✅ Documentation
- `README.md` - User guide
- `INSTALLATION.md` - Installation instructions
- `requirements.txt` - Python dependencies

## 🔧 What's Included

### Preprocessing
- **Image alignment** - Automatic feature-based alignment (built-in)
- **ROI extraction** - Connector cropping based on roi_config.json (built-in)
- **Image normalization** - CLAHE and preprocessing (built-in)

### Models
1. **Occlusion Detection Model** (`OcclusionCNN`)
   - Detects if connector is occluded
   - Input: RGB 128x128 crop
   - Output: OCCLUSION or VISIBLE

2. **PCL Presence Models** (`PCLPresenceClassifier` - ResNet18 based)
   - Binary classification: OK vs KO
   - Input: Grayscale connector crop
   - Output: Probability KO + binary prediction
   - Threshold calibration: Max(OK) + 15% margin method

### All 9 Connectors Trained
- conn1, conn2, conn3, conn4, conn5, conn6, conn7, conn8, conn9
- Each with its own trained model and calibrated threshold

## 🚀 Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run application
python3 beko_pcl_presence_system.py
```

## 📋 System Workflow

1. **Load Image** → User provides PCB image
2. **Align Image** → Automatic alignment with reference template
3. **Extract Connectors** → Crop each connector using ROI config
4. **Step 1: Occlusion Detection** → Check if connector is occluded
5. **Step 2: PCL Presence** → If not occluded, classify OK/KO
6. **Visualize Results** → Display board schematic with status

## 🎯 Key Features

- ✅ **Self-contained**: All models and configs included
- ✅ **No external dependencies**: Preprocessing built-in
- ✅ **Portable**: Can be moved to any location
- ✅ **Complete**: All 9 connectors trained and ready
- ✅ **Updated**: Uses latest threshold calibration method (Max + 15%)

## 📝 Notes

- **Preprocessing scripts**: Not needed - all preprocessing is built into the application
- **Reference images**: Optional - automatic alignment works without them
- **Simulation images**: Optional - create `simulation_images/` folder if needed
- **Logo files**: Optional - application works without them

## 🔄 Updates

To update models:
1. Replace files in `weights/connX/` directories
2. Update `threshold*.json` files
3. Restart application

---

**Package Version**: 1.0  
**Created**: 2026-02-06  
**Status**: ✅ Complete and ready to use
