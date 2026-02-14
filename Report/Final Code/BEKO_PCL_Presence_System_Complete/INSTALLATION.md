# Installation Guide - BEKO PCL Presence Detection System

## Quick Installation

### Step 1: Install Python Dependencies

```bash
pip install -r requirements.txt
```

Or manually install:

```bash
pip install torch torchvision opencv-python pillow numpy matplotlib tkinterdnd2 pygame
```

### Step 2: Verify Package Structure

Ensure the following structure exists:

```
BEKO_PCL_Presence_System_Complete/
├── beko_pcl_presence_system.py
├── models/
│   └── occlusion_cnn.pth
├── weights/
│   ├── conn1/
│   │   ├── model*.pt
│   │   └── threshold*.json
│   └── ... (conn2 through conn9)
├── config/
│   └── roi_config.json
└── requirements.txt
```

### Step 3: Run the Application

```bash
python3 beko_pcl_presence_system.py
```

## Optional: Simulation Images

To enable simulation mode, create a folder with test images:

```bash
mkdir simulation_images
# Add PNG or JPG images to this folder
```

## Troubleshooting

### Import Errors

If you get import errors:
- Ensure Python 3.8+ is installed: `python3 --version`
- Install missing packages: `pip install <package_name>`

### Model Loading Errors

- Verify `models/occlusion_cnn.pth` exists
- Check that `weights/connX/` directories contain model files
- Ensure file permissions allow reading

### GUI Not Opening

- Verify Tkinter is installed: `python3 -m tkinter`
- On Linux, may need: `sudo apt-get install python3-tk`

## System Requirements

- **OS**: Windows, macOS, or Linux
- **Python**: 3.8 or higher
- **RAM**: Minimum 4GB (8GB recommended)
- **GPU**: Optional (CUDA supported, but CPU works fine)

## Verification

After installation, the application should:
1. Load all 9 connector models (conn1 through conn9)
2. Load the occlusion detection model
3. Display "Models loaded (9 connectors)" in the status bar
4. Be ready to analyze images
