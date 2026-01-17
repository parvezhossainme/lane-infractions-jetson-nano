# Improper Speed Detection

Detects vehicles driving too fast or too slow using object tracking and speed estimation.

## Features

- Real-time vehicle tracking
- Speed estimation from pixel movement
- Configurable speed limits (min/max)
- Violation detection and logging
- Visual feedback on video

## Usage

```bash
# Run on video
python demo_speed.py traffic_video.mp4

# With custom output path
python demo_speed.py traffic_video.mp4 output/result.mp4
```

## Configuration

In `speed_detector.py`, adjust:
- `pixels_per_meter`: Calibration for your camera (default: 10.0)
- `min_speed_kmh`: Minimum allowed speed (default: 20 km/h)
- `max_speed_kmh`: Maximum allowed speed (default: 60 km/h)
- `speed_zone`: Zone type - "urban", "highway", "school"

## How It Works

1. **Vehicle Detection**: Uses RT-DETR to detect vehicles
2. **Tracking**: Matches detections across frames
3. **Speed Calculation**: Estimates speed from position changes
4. **Violation Detection**: Compares speed against limits
5. **Visualization**: Annotates video with speeds and warnings

## Calibration

To calibrate `pixels_per_meter`:
1. Measure a known distance in the scene (e.g., 10 meters)
2. Count pixels for that distance
3. `pixels_per_meter = pixels / meters`

## Output

- Annotated video showing vehicle speeds
- Green: Normal speed
- Orange: Too slow
- Red: Speeding
- Console log of all violations
