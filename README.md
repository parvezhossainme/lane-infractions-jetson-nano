# RT-DETR Vehicle and Lane Detection

This project contains modules for vehicle detection using RT-DETR and lane detection using computer vision techniques.

## Project Structure

- `lane_detection.py` - Lane detection module with ROI and Hough transform
- `vehicle_detection.py` - Vehicle detection module using RT-DETR
- `setup_rtdetr.py` - Setup script for installing RT-DETR and dependencies
- `requirements.txt` - Python package requirements

## Setup

### 1. Install Dependencies

First, run the setup script to install RT-DETR and all dependencies:

```bash
python setup_rtdetr.py
```

This will:
- Check GPU availability
- Install PyTorch with CUDA support
- Clone RT-DETR repository
- Download pretrained model weights
- Verify installation

Alternatively, install packages manually:

```bash
pip install -r requirements.txt
```

### 2. Manual RT-DETR Setup (if setup script fails)

```bash
# Install PyTorch with CUDA
pip install torch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 --index-url https://download.pytorch.org/whl/cu121

# Clone RT-DETR
git clone https://github.com/lyuwenyu/RT-DETR.git
cd RT-DETR/rtdetr_pytorch

# Download pretrained model
mkdir checkpoints
wget -P checkpoints https://github.com/lyuwenyu/storage/releases/download/v0.0.1/rtdetr_r50vd_coco.pth

# Install RT-DETR requirements
pip install -r requirements.txt
```

## Usage

### Lane Detection

**Command line:**
```bash
# Activate virtual environment first
source venv/bin/activate

# Run with your image
python lane_detection.py path/to/your/image.jpg

# Or use the helper script
./run_lane_detection.sh path/to/your/image.jpg
```

**In Python code:**
```python
from lane_detection import fixed_roi_for_this_image, detect_lanes
import cv2

# Load image
img = cv2.imread("your_image.jpg")
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# Apply ROI
roi_img, roi_poly = fixed_roi_for_this_image(img)

# Detect lanes
final, lane_img = detect_lanes(img, roi_img)
```

### Vehicle Detection

**Command line:**
```bash
# Activate virtual environment first
source venv/bin/activate

# Setup RT-DETR (one-time setup)
cd RT-DETR/rtdetr_pytorch

# Run with your image
python ../../vehicle_detection.py path/to/your/image.jpg

# Or use the helper script (from project root)
./run_vehicle_detection.sh path/to/your/image.jpg
```

**In Python code:**
```python
from vehicle_detection import VehicleDetector
import cv2

# Load image
img = cv2.imread("your_image.jpg")
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# Initialize detector
detector = VehicleDetector()

# Detect vehicles
boxes, labels, scores = detector.detect(img, confidence_threshold=0.5)

# Visualize
output = detector.visualize(img, boxes, labels, scores)
```

## Requirements

- Python 3.8+
- CUDA-capable GPU (recommended)
- NVIDIA drivers and CUDA toolkit

## Vehicle Classes

The vehicle detector identifies the following COCO classes:
- Class 2: Car
- Class 3: Motorcycle
- Class 5: Bus
- Class 7: Truck

## Notes

- The lane detection uses a fixed ROI optimized for a specific camera view
- Vehicle detection requires RT-DETR to be properly set up in the working directory
- Make sure to update the image path in the demo scripts to match your input images

## Future Enhancements

The code includes a framework outline for:
- Object tracking (assigning IDs to vehicles)
- Speed estimation (using pixel movement)
- Lane violation detection
- Real-time processing
