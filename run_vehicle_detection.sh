#!/bin/bash
# Activate virtual environment and run vehicle detection
# Note: RT-DETR must be set up first using setup_rtdetr.py

if [ $# -eq 0 ]; then
    echo "Usage: ./run_vehicle_detection.sh <image_path>"
    echo "Example: ./run_vehicle_detection.sh IMG_20250813_161947.jpg"
    exit 1
fi

source venv/bin/activate
cd RT-DETR/rtdetr_pytorch
python ../../vehicle_detection.py "$1"
