# Project Structure

```
fydp/
├── detectors/                      # Core detection modules
│   ├── __init__.py
│   ├── lane_detection.py          # Lane detection using Hough transforms
│   ├── lane_detection_save.py     # Lane detection with file output
│   ├── vehicle_detection_ultra.py # Vehicle detection using RT-DETR
│   └── check_setup.py             # Environment verification
│
├── violations/                     # Traffic violation detectors
│   ├── __init__.py
│   ├── speed_detection/           # Speed violation detection
│   │   ├── __init__.py
│   │   ├── speed_detector.py      # Speed tracking and violation detection
│   │   ├── demo_speed.py          # Demo script
│   │   └── README.md
│   ├── lane_violation/            # Lane violation with LSTM
│   │   ├── __init__.py
│   │   ├── lane_violation_detector.py  # LSTM-based detection
│   │   ├── demo_lane_violation.py      # Demo script
│   │   └── README.md
│   └── illegal_stopping/          # Illegal stopping detection
│       ├── __init__.py
│       ├── illegal_stopping_detector.py  # Zone-based detection
│       ├── demo_illegal_stopping.py      # Demo script
│       └── README.md
│
├── tests/                         # Test scripts and samples
│   ├── README.md
│   ├── test_all_detectors.py     # Comprehensive test suite
│   ├── test_speed_detection.py   # Speed detection test
│   ├── test_lane_violation.py    # Lane violation test
│   └── test_illegal_stopping.py  # Illegal stopping test
│
├── samples/                       # Sample data
│   ├── images/                    # Sample images
│   │   ├── IMG_20250813_161947.jpg
│   │   └── lane (1).png
│   └── videos/                    # Sample videos
│       └── traffic_sample.mp4
│
├── outputs/                       # Generated outputs
│   ├── output/                    # Detection results
│   └── output2/                   # Additional outputs
│
├── venv/                          # Python virtual environment
├── RT-DETR/                       # RT-DETR model files
│
├── requirements.txt               # Python dependencies
├── README.md                      # Project documentation
├── QUICKSTART.md                  # Quick start guide
├── .gitignore                     # Git ignore rules
├── run_lane_detection.sh          # Lane detection script
├── run_vehicle_detection.sh       # Vehicle detection script
└── rtdetr-l.pt                    # Pre-trained RT-DETR model
```

## Directory Descriptions

### `detectors/`
Core detection functionality that can be used independently or as building blocks for violation detection.

### `violations/`
Specialized violation detection modules organized by type:
- **speed_detection**: Track vehicle speeds and detect speeding/slow driving
- **lane_violation**: Use LSTM to predict trajectories and detect improper lane changes
- **illegal_stopping**: Zone-based detection of illegal parking/stopping

### `tests/`
Test scripts to verify functionality of all detection modules. Includes comprehensive test suite and individual module tests.

### `samples/`
Sample images and videos for testing and demonstration purposes.

### `outputs/`
Generated output files from detection runs, including annotated images and videos.

## Usage

### Import Detection Modules
```python
# Core detectors
from detectors import detect_lanes, detect_vehicles

# Violation detectors
from violations.speed_detection import SpeedDetector
from violations.lane_violation import LaneViolationDetector
from violations.illegal_stopping import IllegalStoppingDetector, ZoneType
```

### Run Tests
```bash
# Activate virtual environment
source venv/bin/activate

# Run comprehensive test
python tests/test_all_detectors.py

# Run individual tests
python tests/test_speed_detection.py
python tests/test_lane_violation.py
python tests/test_illegal_stopping.py
```

### Run Demos
```bash
# Speed detection on video
python violations/speed_detection/demo_speed.py samples/videos/traffic_sample.mp4

# Lane violation detection
python violations/lane_violation/demo_lane_violation.py samples/videos/traffic_sample.mp4

# Illegal stopping detection
python violations/illegal_stopping/demo_illegal_stopping.py samples/videos/traffic_sample.mp4
```
