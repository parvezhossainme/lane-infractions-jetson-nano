# Illegal Stopping Detection

Detects vehicles illegally stopping or parking in prohibited zones using temporal analysis and zone-based rules.

## Features

- Zone-based stopping regulations (No Stopping, No Parking, Bus Stop, Loading Zone)
- Temporal tracking of stationary vehicles
- Movement analysis to determine if vehicle is stopped
- Configurable time thresholds
- Multiple violation severity levels

## Usage

```bash
# Run on video
python demo_illegal_stopping.py traffic_video.mp4

# With custom output
python demo_illegal_stopping.py traffic_video.mp4 output/result.mp4
```

**Note**: You'll need to adjust zone coordinates in `demo_illegal_stopping.py` to match your video scene.

## Zone Types

1. **NO_STOPPING**: No stopping allowed at all (immediate violation)
2. **NO_PARKING**: Brief stops OK (<3 min), parking not allowed
3. **BUS_STOP**: Only buses can stop
4. **LOADING_ZONE**: Time-limited for loading/unloading
5. **ALLOWED**: Normal parking zones

## Configuration

In `illegal_stopping_detector.py`:
- `movement_threshold`: Pixels to consider as movement (default: 5.0)
- `stationary_time`: Seconds before considered stopped (default: 3.0)

Add zones with:
```python
detector.add_zone(x1, y1, x2, y2, ZoneType.NO_STOPPING, max_duration=0)
```

## How It Works

1. **Vehicle Tracking**: Tracks each vehicle's position over time
2. **Movement Analysis**: Calculates movement between frames
3. **Stationary Detection**: Identifies when vehicle stops moving
4. **Zone Matching**: Determines which regulation zone vehicle is in
5. **Violation Detection**: Checks if stop duration violates zone rules
6. **Temporal Filtering**: Only reports violations after minimum stop time

## Violation Types

- `NO_STOPPING_ZONE_VIOLATION`: Stopped in no-stopping zone (HIGH)
- `NO_PARKING_VIOLATION`: Parked too long in no-parking zone (MEDIUM)
- `BUS_STOP_VIOLATION`: Non-bus vehicle in bus stop (MEDIUM)
- `LOADING_ZONE_VIOLATION`: Exceeded loading zone time (LOW)
- `SUSPICIOUS_STOPPING`: Unexplained long stop (LOW)

## Customization

### Define Custom Zones

Edit `demo_illegal_stopping.py` to add zones matching your scene:

```python
# No stopping zone on left side
detector.add_zone(0, 0, 200, 480, ZoneType.NO_STOPPING, 0)

# No parking with 3 minute grace period
detector.add_zone(200, 0, 400, 480, ZoneType.NO_PARKING, 180)

# Bus stop - only buses, 1 min max
detector.add_zone(400, 0, 600, 480, ZoneType.BUS_STOP, 60)
```

### Adjust Sensitivity

- Increase `movement_threshold` if too many false positives
- Decrease `stationary_time` to catch shorter stops
- Adjust zone `max_duration` for each regulation type

## Visualization

- **Colored zones**: Overlaid on video
  - Red: No Stopping
  - Orange: No Parking
  - Cyan: Bus Stop
  - Yellow: Loading Zone
  - Green: Parking Allowed
- **Stopped vehicles**: Marked with yellow circles and stop duration
- **Violations**: Red warning boxes with violation type

## Future Enhancements

- Automatic zone detection from road markings
- License plate recognition for violation logging
- Emergency vehicle exemptions
- Weather-based regulation adjustments
- Integration with parking payment systems
