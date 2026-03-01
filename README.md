# Traffic Violations Detection (Simple Guide)

This project detects vehicles, lanes, and traffic violations (speeding, lane violation, and illegal stopping) using RT-DETR plus OpenCV-based processing.

## 1) Setup

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Model file location:

- `models/rtdetr-l.pt`

## 2) Main Folders

- `detectors/` core lane and vehicle detection modules
- `violations/` violation-specific detectors and demos
- `scripts/metrics/` metric/report generation scripts
- `samples/` sample input images/videos
- `outputs/` generated outputs and reports
- `tests/` test scripts

## 3) Quick Run Examples

```bash
# lane detection (image)
python lane_detection_hough.py samples/images/IMG_20250813_161947.jpg

# vehicle detection (image)
python vehicle_detection.py samples/images/IMG_20250813_161947.jpg

# vehicle detection (video)
python detectors/vehicle_detection_video.py --input samples/videos/Sample-Main.mp4 --save
```

## 4) Metrics Scripts

```bash
python scripts/metrics/speed_metric.py
python scripts/metrics/lane_violation_metric.py
python scripts/metrics/illegal_stopping_metric.py
python scripts/metrics/generate_paper_metrics.py
```

## 5) Tests

```bash
python tests/test_all_detectors.py
python tests/test_speed_detection.py
python tests/test_lane_violation.py
python tests/test_illegal_stopping.py
```

## Notes

- Run commands from the project root folder.
- If you changed local file locations, update default paths in scripts accordingly.
