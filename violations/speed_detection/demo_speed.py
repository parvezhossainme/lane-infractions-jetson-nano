"""
Speed Detection Demo
Test speed detection on video or images.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import cv2
import numpy as np
from violations.speed_detection import SpeedDetector
from ultralytics import RTDETR


def run_speed_detection(video_path: str, output_path: str = "output/speed_detection.mp4"):
    """
    Run speed detection on video.
    
    Args:
        video_path: Path to input video
        output_path: Path to save output video
    """
    # Initialize detector and model
    print("Loading RT-DETR model...")
    model = RTDETR('models/rtdetr-l.pt')
    
    print("Initializing speed detector...")
    speed_detector = SpeedDetector(
        pixels_per_meter=10.0,  # Calibrate this for your camera
        fps=30.0,
        min_speed_kmh=20.0,
        max_speed_kmh=60.0,  # Urban speed limit
        speed_zone="urban"
    )
    
    # Open video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video: {video_path}")
        return
    
    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"Video: {width}x{height} @ {fps} FPS, {total_frames} frames")
    
    # Setup output
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    frame_number = 0
    all_violations = []
    
    print("\nProcessing video...")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Detect vehicles
        results = model(frame, conf=0.5, verbose=False)
        
        # Extract detections
        detections = []
        for box in results[0].boxes:
            cls_id = int(box.cls[0])
            # Filter for vehicles only (car, motorcycle, bus, truck)
            if cls_id in [2, 3, 5, 7]:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                detections.append({
                    'bbox': [x1, y1, x2, y2],
                    'class_id': cls_id,
                    'confidence': float(box.conf[0])
                })
        
        # Update speed detector
        violations = speed_detector.update(detections, frame_number)
        all_violations.extend(violations)
        
        # Visualize
        output_frame = speed_detector.visualize(frame, violations)
        
        # Write frame
        out.write(output_frame)
        
        # Progress
        if frame_number % 30 == 0:
            progress = (frame_number / total_frames) * 100
            print(f"Progress: {progress:.1f}% - Frame {frame_number}/{total_frames} - Violations: {len(all_violations)}")
        
        frame_number += 1
    
    cap.release()
    out.release()
    
    # Print summary
    print("\n" + "="*60)
    print("Speed Detection Summary")
    print("="*60)
    print(f"Total frames processed: {frame_number}")
    print(f"Total violations detected: {len(all_violations)}")
    
    # Count violation types
    speeding_count = sum(1 for v in all_violations if v['type'] == 'SPEEDING')
    too_slow_count = sum(1 for v in all_violations if v['type'] == 'TOO_SLOW')
    
    print(f"  - Speeding violations: {speeding_count}")
    print(f"  - Too slow violations: {too_slow_count}")
    
    print(f"\nOutput saved to: {output_path}")
    print("="*60)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python demo_speed.py <video_path> [output_path]")
        print("Example: python demo_speed.py traffic.mp4 output/speed_result.mp4")
        sys.exit(1)
    
    video_path = sys.argv[1]
    output_path = sys.argv[2] if len(sys.argv) > 2 else "output/speed_detection.mp4"
    
    run_speed_detection(video_path, output_path)
