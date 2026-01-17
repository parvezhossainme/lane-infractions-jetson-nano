"""
Illegal Stopping Detection Demo
Test illegal stopping detection on video.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2
import numpy as np
from illegal_stopping_detector import IllegalStoppingDetector, ZoneType
from ultralytics import RTDETR


def run_illegal_stopping_detection(video_path: str, output_path: str = "output/illegal_stopping.mp4"):
    """
    Run illegal stopping detection on video.
    
    Args:
        video_path: Path to input video
        output_path: Path to save output
    """
    # Initialize
    print("Loading RT-DETR model...")
    model = RTDETR('rtdetr-l.pt')
    
    print("Initializing illegal stopping detector...")
    detector = IllegalStoppingDetector(
        movement_threshold=5.0,
        stationary_time=3.0
    )
    
    # Define stopping zones (example - adjust for your scene)
    # Format: (x1, y1, x2, y2, zone_type, max_duration)
    print("Setting up regulation zones...")
    
    # Example zones (adjust coordinates for your video)
    detector.add_zone(100, 200, 300, 400, ZoneType.NO_STOPPING, 0)
    detector.add_zone(400, 200, 600, 400, ZoneType.NO_PARKING, 180)  # 3 min max
    detector.add_zone(700, 200, 900, 400, ZoneType.BUS_STOP, 60)  # 1 min for loading
    
    # Open video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video: {video_path}")
        return
    
    # Video properties
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
    print("Note: Violations are detected after vehicles stop for 3+ seconds")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        timestamp = frame_number / fps
        
        # Detect vehicles
        results = model(frame, conf=0.5, verbose=False)
        
        # Extract detections
        detections = []
        for box in results[0].boxes:
            cls_id = int(box.cls[0])
            if cls_id in [2, 3, 5, 7]:  # Vehicles
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                detections.append({
                    'bbox': [x1, y1, x2, y2],
                    'class_id': cls_id,
                    'confidence': float(box.conf[0])
                })
        
        # Update detector
        violations = detector.update(detections, timestamp)
        all_violations.extend(violations)
        
        # Visualize
        output_frame = detector.visualize(frame, violations)
        
        # Write frame
        out.write(output_frame)
        
        # Progress
        if frame_number % 30 == 0:
            progress = (frame_number / total_frames) * 100
            stopped_vehicles = sum(1 for v in detector.vehicles.values() 
                                  if detector.is_vehicle_stationary(v))
            print(f"Progress: {progress:.1f}% - Frame {frame_number}/{total_frames} - "
                  f"Stopped: {stopped_vehicles} - Violations: {len(all_violations)}")
        
        frame_number += 1
    
    cap.release()
    out.release()
    
    # Summary
    print("\n" + "="*60)
    print("Illegal Stopping Detection Summary")
    print("="*60)
    print(f"Total frames processed: {frame_number}")
    print(f"Total violations detected: {len(all_violations)}")
    
    # Count by type
    violation_types = {}
    for v in all_violations:
        vtype = v['type']
        violation_types[vtype] = violation_types.get(vtype, 0) + 1
    
    for vtype, count in violation_types.items():
        print(f"  - {vtype}: {count}")
    
    print(f"\nOutput saved to: {output_path}")
    print("="*60)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python demo_illegal_stopping.py <video_path> [output_path]")
        print("Example: python demo_illegal_stopping.py traffic.mp4 output/stopping_result.mp4")
        print("\nNote: You may need to adjust zone coordinates in the script for your video.")
        sys.exit(1)
    
    video_path = sys.argv[1]
    output_path = sys.argv[2] if len(sys.argv) > 2 else "output/illegal_stopping.mp4"
    
    run_illegal_stopping_detection(video_path, output_path)
