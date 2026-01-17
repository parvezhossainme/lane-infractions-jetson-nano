# Quick Start Guide

## ✅ Setup Complete!

Your Jupyter notebook has been converted to Python files with a virtual environment set up.

## Project Files

### Core Modules
- **lane_detection.py** - Interactive lane detection (shows plots)
- **lane_detection_save.py** - Saves lane detection results to files
- **vehicle_detection.py** - Vehicle detection using RT-DETR
- **setup_rtdetr.py** - RT-DETR installation script

### Configuration
- **requirements.txt** - Python dependencies
- **venv/** - Virtual environment (already set up)
- **.vscode/settings.json** - VS Code Python interpreter config

### Helper Scripts
- **run_lane_detection.sh** - Quick lane detection runner
- **run_vehicle_detection.sh** - Quick vehicle detection runner

## Usage Examples

### 1. Lane Detection (Save Results)

```bash
# Activate virtual environment
source venv/bin/activate

# Run lane detection - saves images to output/
python lane_detection_save.py IMG_20250813_161947.jpg

# Or specify custom output directory
python lane_detection_save.py IMG_20250813_161947.jpg my_results/
```

**Output:**
- `output/1_original.png` - Original image
- `output/2_roi.png` - Region of interest visualization
- `output/3_lane_binary.png` - Preprocessed binary image
- `output/4_final_result.png` - Final lane detection result

### 2. Lane Detection (Interactive)

```bash
source venv/bin/activate
python lane_detection.py IMG_20250813_161947.jpg
```

### 3. Vehicle Detection with RT-DETR

**First-time setup:**
```bash
source venv/bin/activate
python setup_rtdetr.py
```

**Run detection:**
```bash
source venv/bin/activate
cd RT-DETR/rtdetr_pytorch
python ../../vehicle_detection.py ../../IMG_20250813_161947.jpg
```

## Available Images

You have these images in your directory:
- `IMG_20250813_161947.jpg`
- `IMG_20250813_161847.jpg`
- `lane (1).png`

## Python API Usage

```python
# Import modules
from lane_detection import fixed_roi_for_this_image, detect_lanes
import cv2

# Load image
img = cv2.imread("IMG_20250813_161947.jpg")
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# Process
roi_img, roi_poly = fixed_roi_for_this_image(img)
final, lane_img, lane_binary = detect_lanes(img, roi_img)

# Save result
cv2.imwrite("result.jpg", cv2.cvtColor(final, cv2.COLOR_RGB2BGR))
```

## Notes

- **Virtual Environment**: Always activate with `source venv/bin/activate` before running scripts
- **VS Code**: The Python interpreter is already configured to use the virtual environment
- **matplotlib warnings**: Normal when running in non-interactive mode - use `lane_detection_save.py` to save results instead
- **RT-DETR**: Requires one-time setup with `setup_rtdetr.py` before first use

## Troubleshooting

**"Image not found" error:**
- Use relative paths: `python lane_detection_save.py IMG_20250813_161947.jpg`
- Or absolute paths: `python lane_detection_save.py /full/path/to/image.jpg`

**"Module not found" error:**
- Activate virtual environment: `source venv/bin/activate`

**RT-DETR not working:**
- Run setup first: `python setup_rtdetr.py`
- Make sure you're in the correct directory when running vehicle detection
