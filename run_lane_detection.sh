#!/bin/bash
# Activate virtual environment and run lane detection

if [ $# -eq 0 ]; then
    echo "Usage: ./run_lane_detection.sh <image_path>"
    echo "Example: ./run_lane_detection.sh IMG_20250813_161947.jpg"
    exit 1
fi

source venv/bin/activate
python lane_detection.py "$1"
