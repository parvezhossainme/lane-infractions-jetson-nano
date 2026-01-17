# Test Samples for Traffic Violation Detection

This directory contains test scripts to verify all detection modules.

## Quick Test - All Modules

Run comprehensive test of all detectors:

```bash
cd /home/parvezdev/fydp
source venv/bin/activate
python test_samples/test_all_detectors.py
```

This will:
- ✅ Test vehicle detection with RT-DETR
- ✅ Test lane detection
- ✅ Test speed detection module
- ✅ Test lane violation detection with LSTM
- ✅ Test illegal stopping detection
- ✅ Run integration test with real image

## Individual Module Tests

### 1. Speed Detection Test

Creates synthetic video to test speed tracking:

```bash
python test_samples/test_speed_detection.py
```

Output: `test_samples/output/speed_test.mp4`
- Shows vehicles moving at different speeds
- Displays speed violations (too fast/too slow)
- 5 second test video

### 2. Lane Violation Detection Test

Tests LSTM-based trajectory prediction:

```bash
python test_samples/test_lane_violation.py
```

Output: `test_samples/output/lane_violation_test.mp4`
- Simulates sudden lane changes
- LSTM predicts vehicle trajectories
- Detects deviations from predicted paths
- 8 second test video

### 3. Illegal Stopping Detection Test

Tests zone-based stopping regulations:

```bash
python test_samples/test_illegal_stopping.py
```

Output: `test_samples/output/illegal_stopping_test.mp4`
- Shows 4 different zone types
- Simulates stationary vehicles
- Detects violations based on stop duration
- 10 second test video

## Test Outputs

All test outputs are saved to `test_samples/output/`:
- `vehicle_detection_test.jpg` - Vehicle detection results
- `lane_detection_test.jpg` - Lane detection results
- `speed_test.mp4` - Speed violation detection video
- `lane_violation_test.mp4` - Lane change violation video
- `illegal_stopping_test.mp4` - Illegal stopping video

## Requirements

Make sure virtual environment is activated and dependencies installed:

```bash
source venv/bin/activate
pip install -r requirements.txt
```

## Sample Data

Tests use existing sample images:
- `IMG_20250813_161947.jpg` - Traffic scene with multiple vehicles

## Expected Results

### Vehicle Detection
- Should detect 15-25 vehicles (cars, motorcycles, buses, trucks)

### Lane Detection
- May not detect lanes in all images (depends on road marking visibility)
- Normal if no lanes detected in some test images

### Speed Detection
- Tracks multiple vehicles simultaneously
- Calculates speed from pixel movement
- Reports speeding/too slow violations

### Lane Violation (LSTM)
- Builds 10-frame trajectory sequences
- Predicts next positions using LSTM
- Detects sudden deviations

### Illegal Stopping
- Tracks stationary vehicles
- Matches vehicles to regulation zones
- Reports violations after minimum stop time (3 seconds)

## Troubleshooting

### Import Errors
Make sure you're in the correct directory:
```bash
cd /home/parvezdev/fydp
```

### Module Not Found
Activate virtual environment:
```bash
source venv/bin/activate
```

### GPU Out of Memory
RT-DETR uses GPU by default. If issues occur:
- Close other applications
- Or modify detector to use CPU: `model = RTDETR('rtdetr-l.pt', device='cpu')`

### Video Codec Issues
If mp4v doesn't work, try:
- Install ffmpeg: `sudo apt install ffmpeg`
- Or change fourcc to `'XVID'` or `'H264'`

## Next Steps

After running tests:
1. Review output videos to understand each detector
2. Adjust parameters for your specific use case
3. Test on real traffic videos
4. Calibrate `pixels_per_meter` for accurate speed measurement
5. Define regulation zones based on your location
