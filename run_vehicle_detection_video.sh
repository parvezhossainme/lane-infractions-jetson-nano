#!/bin/bash
# Run vehicle detection on video (separate from image detector flow)

source venv/bin/activate

VIDEO_PATH="${1:-samples/videos/Sample-Main.mp4}"
CONFIDENCE="${2:-0.5}"

python detectors/vehicle_detection_video.py \
  --input "$VIDEO_PATH" \
  --confidence "$CONFIDENCE" \
  --output-dir outputs/output
