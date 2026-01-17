# Driving Over Lane Detection

Detects vehicles improperly crossing lane lines using LSTM trajectory prediction and lane tracking.

## Features

- Lane boundary detection
- Vehicle trajectory tracking with LSTM
- Improper lane change detection (without indicator)
- Lane deviation detection
- Trajectory prediction for anomaly detection

## Usage

```bash
# Run on video
python demo_lane_violation.py traffic_video.mp4

# With custom output
python demo_lane_violation.py traffic_video.mp4 output/result.mp4
```

## How It Works

1. **Lane Detection**: Detects lane boundaries using computer vision
2. **Vehicle Tracking**: Tracks vehicle positions across frames
3. **LSTM Prediction**: Predicts expected trajectory using LSTM neural network
4. **Anomaly Detection**: Compares actual vs predicted trajectory
5. **Violation Detection**: Identifies improper lane changes and deviations

## LSTM Model

The trajectory LSTM:
- Input: Sequence of (x, y) positions (default: last 10 frames)
- Hidden layers: 2 LSTM layers with 64 units each
- Output: Predicted next (x, y) position

Deviations beyond threshold indicate potential lane violations.

## Configuration

In `lane_violation_detector.py`:
- `num_lanes`: Number of lanes (default: 3)
- `sequence_length`: LSTM sequence length (default: 10)
- `deviation_threshold`: Pixels for anomaly detection (default: 50.0)

## Violation Types

1. **IMPROPER_LANE_CHANGE**: Changing lanes without indicator (HIGH severity)
2. **LANE_DEVIATION**: Unusual trajectory deviation (MEDIUM severity)

## Future Enhancements

- Blinker detection using computer vision
- Multi-camera perspective fusion
- Real-time indicator signal detection
- Weather-adaptive thresholds
