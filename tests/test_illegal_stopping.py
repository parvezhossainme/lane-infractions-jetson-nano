"""
Illegal Stopping Detection Test
Simulates stationary vehicles in different zones.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2
import numpy as np
from violations.illegal_stopping import IllegalStoppingDetector, ZoneType
from ultralytics import RTDETR


def create_stopping_test(image_path: str, output_path: str = "outputs/tests/illegal_stopping_test.mp4"):
    """
    Create test video showing illegal stopping detection.
    """
    print("Creating illegal stopping detection test video...")
    
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
    detector = IllegalStoppingDetector(
        movement_threshold=5.0,
        stationary_time=3.0
    )
    
    # Define test zones (split image into regions)
    zone_width = width // 4
    detector.add_zone(0, 0, zone_width, height, ZoneType.NO_STOPPING, 0)
    detector.add_zone(zone_width, 0, zone_width*2, height, ZoneType.NO_PARKING, 180)
    detector.add_zone(zone_width*2, 0, zone_width*3, height, ZoneType.BUS_STOP, 60)
    detector.add_zone(zone_width*3, 0, width, height, ZoneType.ALLOWED, 0)
    
    print(f"Defined {len(detector.zones)} regulation zones")
    
    # Simulate stopped vehicles
    num_frames = fps * 10  # 10 seconds
    total_violations = []
    
    for frame_num in range(num_frames):
        timestamp = frame_num / fps
        
        # Create frame copy
        frame = image.copy()
        
        # Some vehicles stop, others keep moving
        current_detections = []
        for i, det in enumerate(initial_detections):
            x1, y1, x2, y2 = det['bbox']
            
            # First 2 vehicles stop (stationary)
            if i < 2:
                # Stationary - no movement
                current_detections.append({
                    'bbox': [x1, y1, x2, y2],
                    'class_id': det['class_id'],
                    'confidence': det['confidence']
                })
            else:
                # Moving vehicles
                offset_x = 1.0 * frame_num
                new_x1 = min(x1 + offset_x, width - (x2 - x1))
                new_x2 = min(x2 + offset_x, width)
                
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
        stopped_count = sum(1 for v in detector.vehicles.values() 
                          if detector.is_vehicle_stationary(v))
        
        cv2.putText(output_frame, f"Frame: {frame_num}/{num_frames}", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(output_frame, f"Time: {timestamp:.1f}s", (10, 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(output_frame, f"Stopped: {stopped_count}", (10, 90),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        cv2.putText(output_frame, f"Violations: {len(total_violations)}", (10, 120),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        out.write(output_frame)
    
    out.release()
    
    # Print summary
    print("\n" + "="*60)
    print("ILLEGAL STOPPING DETECTION TEST SUMMARY")
    print("="*60)
    print(f"Processed: {num_frames} frames ({num_frames/fps:.1f} seconds)")
    print(f"Tracked: {len(detector.vehicles)} vehicles")
    print(f"Total violations: {len(total_violations)}")
    
    # Count by zone type
    violation_types = {}
    for v in total_violations:
        vtype = v['type']
        violation_types[vtype] = violation_types.get(vtype, 0) + 1
    
    for vtype, count in violation_types.items():
        print(f"  - {vtype}: {count}")
    
    print(f"\nOutput saved: {output_path}")
    print("="*60)


if __name__ == "__main__":
    image_path = "/home/parvezdev/fydp/samples/images/IMG_20250813_161947.jpg"
    
    if not os.path.exists(image_path):
        print(f"Error: Sample image not found: {image_path}")
        sys.exit(1)
    
    create_stopping_test(image_path)
    print("\n[OK] Illegal stopping detection test complete!")
    print("The video shows different zone types and violation detection.")
