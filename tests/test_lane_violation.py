"""
Lane Violation Detection Test with LSTM
Creates synthetic trajectories to test lane change detection.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2
import numpy as np
from violations.lane_violation import LaneViolationDetector
from ultralytics import RTDETR


def create_lane_violation_test(image_path: str, output_path: str = "outputs/tests/lane_violation_test.mp4"):
    """
    Create test video showing lane violation detection with LSTM.
    """
    print("Creating lane violation test video with LSTM predictions...")
    
    # Load image and detect vehicles
    image = cv2.imread(image_path)
    model = RTDETR('models/rtdetr-l.pt')
    results = model(image, conf=0.5, verbose=False)
    
    # Extract initial detections
    initial_detections = []
    for box in results[0].boxes:
        cls_id = int(box.cls[0])
        if cls_id in [2, 3, 5, 7]:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            initial_detections.append({
                'bbox': [x1, y1, x2, y2],
                'class_id': cls_id,
                'confidence': float(box.conf[0])
            })
    
    print(f"Found {len(initial_detections)} vehicles in image")
    
    # Create video
    fps = 30
    width, height = image.shape[1], image.shape[0]
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    # Initialize detector
    detector = LaneViolationDetector(
        sequence_length=10,
        deviation_threshold=50.0
    )
    
    # Simulate movement with lane changes
    num_frames = fps * 8  # 8 seconds
    total_violations = []
    
    for frame_num in range(num_frames):
        timestamp = frame_num / fps
        
        # Create frame copy
        frame = image.copy()
        
        # Simulate vehicle movements
        current_detections = []
        for i, det in enumerate(initial_detections):
            x1, y1, x2, y2 = det['bbox']
            
            # Normal forward movement
            offset_x = 2.0 * frame_num
            
            # Add lane change for some vehicles
            offset_y = 0
            if i % 2 == 0 and frame_num > 60 and frame_num < 120:
                # Sudden lane change (violation)
                offset_y = (frame_num - 60) * 3.0
            elif i % 3 == 0 and frame_num > 150:
                # Gradual lane change (normal)
                offset_y = (frame_num - 150) * 0.5
            
            new_x1 = min(x1 + offset_x, width - (x2 - x1))
            new_x2 = min(x2 + offset_x, width)
            new_y1 = max(0, min(y1 + offset_y, height - (y2 - y1)))
            new_y2 = max(0, min(y2 + offset_y, height))
            
            if new_x1 < width:
                current_detections.append({
                    'bbox': [new_x1, new_y1, new_x2, new_y2],
                    'class_id': det['class_id'],
                    'confidence': det['confidence']
                })
        
        # Update detector
        violations = detector.update(current_detections, frame_num, timestamp)
        total_violations.extend(violations)
        
        # Visualize
        output_frame = detector.visualize(frame, violations)
        
        # Add info
        cv2.putText(output_frame, f"Frame: {frame_num}/{num_frames}", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(output_frame, f"LSTM Prediction Active", (10, 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(output_frame, f"Violations: {len(total_violations)}", (10, 90),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        out.write(output_frame)
    
    out.release()
    
    # Print summary
    print("\n" + "="*60)
    print("LANE VIOLATION DETECTION TEST SUMMARY (LSTM)")
    print("="*60)
    print(f"Processed: {num_frames} frames ({num_frames/fps:.1f} seconds)")
    print(f"Tracked: {len(detector.vehicles)} vehicles")
    print(f"Total violations: {len(total_violations)}")
    
    # Count by type
    lane_changes = sum(1 for v in total_violations if v['type'] == 'SUDDEN_LANE_CHANGE')
    deviations = sum(1 for v in total_violations if v['type'] == 'PATH_DEVIATION')
    
    print(f"  - Sudden lane changes: {lane_changes}")
    print(f"  - Path deviations: {deviations}")
    print(f"\nOutput saved: {output_path}")
    print("="*60)


if __name__ == "__main__":
    image_path = "/home/parvezdev/fydp/samples/images/IMG_20250813_161947.jpg"
    
    if not os.path.exists(image_path):
        print(f"Error: Sample image not found: {image_path}")
        sys.exit(1)
    
    create_lane_violation_test(image_path)
    print("\n[OK] Lane violation detection test complete!")
    print("The LSTM model predicted trajectories and detected deviations.")
