"""
Speed Detection Test with Sample Image
Creates a synthetic video from static image to test speed tracking.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2
import numpy as np
from violations.speed_detection import SpeedDetector
from ultralytics import RTDETR


def create_synthetic_video(image_path: str, output_path: str = "outputs/tests/speed_test.mp4"):
    """
    Create synthetic video by moving detected vehicles to simulate motion.
    """
    print("Creating synthetic video for speed detection test...")
    
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
    detector = SpeedDetector(
        pixels_per_meter=10.0,  # Adjust based on your camera
        fps=fps,
        min_speed_kmh=20.0,
        max_speed_kmh=60.0
    )
    
    # Simulate movement for 5 seconds
    num_frames = fps * 5
    total_violations = []
    
    for frame_num in range(num_frames):
        timestamp = frame_num / fps
        
        # Create frame copy
        frame = image.copy()
        
        # Move vehicles (simulate different speeds)
        current_detections = []
        for i, det in enumerate(initial_detections):
            # Different vehicles move at different speeds
            speed_factor = 2 + i * 1.5  # pixels per frame
            
            x1, y1, x2, y2 = det['bbox']
            offset = speed_factor * frame_num
            
            # Move horizontally
            new_x1 = min(x1 + offset, width - (x2 - x1))
            new_x2 = min(x2 + offset, width)
            
            if new_x1 < width:
                current_detections.append({
                    'bbox': [new_x1, y1, new_x2, y2],
                    'class_id': det['class_id'],
                    'confidence': det['confidence']
                })
        
        # Update detector
        violations = detector.update(current_detections, timestamp)
        total_violations.extend(violations)
        
        # Visualize
        output_frame = detector.visualize(frame, violations)
        
        # Add info
        cv2.putText(output_frame, f"Frame: {frame_num}/{num_frames}", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(output_frame, f"Time: {timestamp:.1f}s", (10, 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(output_frame, f"Violations: {len(total_violations)}", (10, 90),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        out.write(output_frame)
    
    out.release()
    
    # Print summary
    print("\n" + "="*60)
    print("SPEED DETECTION TEST SUMMARY")
    print("="*60)
    print(f"Processed: {num_frames} frames ({num_frames/fps:.1f} seconds)")
    print(f"Tracked: {len(initial_detections)} vehicles")
    print(f"Total violations: {len(total_violations)}")
    
    # Count by type
    speeding = sum(1 for v in total_violations if v['type'] == 'SPEEDING')
    too_slow = sum(1 for v in total_violations if v['type'] == 'TOO_SLOW')
    
    print(f"  - Speeding: {speeding}")
    print(f"  - Too slow: {too_slow}")
    print(f"\nOutput saved: {output_path}")
    print("="*60)


if __name__ == "__main__":
    image_path = "/home/parvezdev/fydp/samples/images/IMG_20250813_161947.jpg"
    
    if not os.path.exists(image_path):
        print(f"Error: Sample image not found: {image_path}")
        sys.exit(1)
    
    create_synthetic_video(image_path)
    print("\n[OK] Speed detection test complete!")
    print("View the output video to see speed tracking in action.")
