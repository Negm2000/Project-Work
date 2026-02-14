# BEKO PCL Presence Detection System - Complete Package

Complete standalone package for PCB connector analysis using PCL Presence Detection models.

## 📁 Package Structure

```
BEKO_PCL_Presence_System_Complete/
├── beko_pcl_presence_system.py    # Main GUI application
├── models/
│   └── occlusion_cnn.pth          # Occlusion detection model
├── weights/
│   ├── conn1/                      # Connector 1 model
│   │   ├── model.pt
│   │   └── threshold.json
│   ├── conn2/                      # Connector 2 model
│   │   ├── model (1).pt
│   │   └── threshold (1).json
│   ├── conn3/                      # Connector 3 model
│   │   ├── model (2).pt
│   │   └── threshold (2).json
│   ├── ...                         # conn4 through conn9
│   └── conn9/
│       ├── model (8).pt
│       └── threshold (8).json
├── config/
│   └── roi_config.json            # ROI configuration for connectors
├── requirements.txt                # Python dependencies
├── README.md                       # This file
└── INSTALLATION.md                 # Installation guide
```

**Note**: All paths are relative to this package directory. The software is completely self-contained.

## 🚀 Quick Start

### Prerequisites

Install required Python packages:

```bash
pip install torch torchvision opencv-python pillow numpy matplotlib tkinterdnd2 pygame
```

Or use the requirements file:

```bash
pip install -r requirements.txt
```

### Running the Application

```bash
cd BEKO_PCL_Presence_System_Complete
python3 beko_pcl_presence_system.py
```

## 📋 Features

- **Occlusion Detection**: First step to identify occluded connectors
- **PCL Presence Detection**: Binary classification (OK/KO) for each connector
- **Interactive GUI**: Drag-and-drop interface for image analysis
- **Simulation Mode**: Automated testing on multiple images
- **Visual Feedback**: Board schematic with connector status and enlarged crops

## 🔧 System Requirements

- Python 3.8+
- PyTorch (CPU or CUDA)
- OpenCV
- Tkinter (usually included with Python)
- TkinterDnD2 (for drag-and-drop)
- Matplotlib
- NumPy
- Pillow

## 📖 Usage

### Basic Analysis

1. Launch the application
2. Drag and drop a PCB image onto the interface, or click "Select Image"
3. Click "ANALYZE IMAGE"
4. View results in the main display area

### Simulation Mode

1. Create a `simulation_images/` folder in the package directory
2. Add test images (PNG or JPG) to this folder
3. Click "Start" in the Simulation panel
4. The system will automatically process images from the queue

## 🎯 Models Included

### Occlusion Detection Model
- **File**: `models/occlusion_cnn.pth`
- **Purpose**: Detects if a connector is occluded (blocked by cables, etc.)
- **Input**: RGB connector crop (128x128)
- **Output**: OCCLUSION or VISIBLE

### PCL Presence Models
- **Files**: `weights/connX/model*.pt` and `threshold*.json`
- **Purpose**: Binary classification for PCL presence (OK/KO)
- **Input**: Grayscale connector crop
- **Output**: Probability of KO + binary prediction based on threshold

### Threshold Calibration
- Thresholds are calibrated using the **Max(OK) + 15% margin** method
- Each connector has its own threshold stored in `threshold*.json`

## 📝 Configuration

### ROI Configuration
The `config/roi_config.json` file defines the rectangular regions of interest (ROI) for each connector on the PCB board.

### Model Weights
All trained models are stored in the `weights/` directory, organized by connector:
- `conn1/` through `conn9/`
- Each directory contains:
  - `model*.pt`: Trained PyTorch model weights
  - `threshold*.json`: Calibrated threshold for binary classification

## 🔍 Troubleshooting

### Models Not Loading
- Verify that `models/occlusion_cnn.pth` exists
- Check that `weights/connX/` directories contain `model*.pt` and `threshold*.json` files
- Ensure file permissions allow reading

### Image Alignment Issues
- The system uses automatic alignment based on feature matching
- If alignment fails, ensure images are similar to the reference template
- Reference images can be placed in `reference/` folder (optional)

### Simulation Not Working
- Create `simulation_images/` folder in package directory
- Add PNG or JPG images to this folder
- Ensure images are valid PCB images

## 📊 Model Performance

Each connector model has been trained and calibrated:
- **Training Data**: Real OK samples + synthetic KO samples
- **Validation**: Real OK samples only
- **Threshold Method**: Max(OK probability) + 15% margin
- **Expected FPR**: ~0% on OK samples (very conservative)

## 🔄 Updates

To update models:
1. Replace model files in `weights/connX/` directories
2. Update corresponding `threshold*.json` files
3. Restart the application

## 📧 Support

For issues or questions, refer to the main project documentation in `Project_Work_Complete/`.

## 📄 License

This software is part of the BEKO PCL Presence Detection project.

---

**Version**: 1.0  
**Last Updated**: 2026-02-06  
**Package**: Complete standalone system with all models and dependencies
