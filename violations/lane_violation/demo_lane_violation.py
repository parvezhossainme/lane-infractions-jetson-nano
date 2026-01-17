"""
Lane Violation Detection Demo
Test lane violation detection on video.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import cv2
import numpy as np
from violations.lane_violation import LaneViolationDetector
from ultralytics import RTDETR


def detect_lane_lines(frame: np.ndarray) -> tuple:
    """
    Simple lane detection (reuse from existing lane_detection module).
    
    Args:
        frame: Input frame
        
    Returns:
        (lane_image, lane_boundaries)
    """
    # Import lane detection from parent directory
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from lane_detection import preprocess_lane, fixed_roi_for_this_image
    
    # Apply ROI
    roi_img, _ = fixed_roi_for_this_image(frame)
    
    # Preprocess
    lane_binary = preprocess_lane(roi_img)
    
    # Detect edges
    edges = cv2.Canny(lane_binary, 50, 150)
    
    # Hough lines
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, 100, minLineLength=100, maxLineGap=50)
    
    # Create lane visualization
    lane_img = np.zeros_like(frame)
    lane_boundaries = []
    
    if lines is not None:
        # Group lines by x-position to find lane boundaries
        x_positions = []
        for line in lines:
            x1, y1, x2, y2 = line[0]
            # Filter near-vertical lines (lane markings)
            angle = abs(np.degrees(np.arctan2(y2-y1, x2-x1)))
            if 60 < angle < 120:
                cv2.line(lane_img, (x1, y1), (x2, y2), (0, 255, 0), 3)
                x_positions.extend([x1, x2])
        
        # Find lane boundaries (cluster x-positions)
        if x_positions:
            x_positions = sorted(x_positions)
            # Simple clustering - group nearby x values
            boundaries = [x_positions[0]]
            for x in x_positions:
                if x - boundaries[-1] > 100:  # Minimum distance between lanes
                    boundaries.append(x)
            lane_boundaries = boundaries
    
    return lane_img, lane_boundaries


def run_lane_violation_detection(video_path: str, output_path: str = "output/lane_violations.mp4"):
    """
    Run lane violation detection on video.
    
    Args:
        video_path: Path to input video
        output_path: Path to save output
    """
    # Initialize
    print("Loading RT-DETR model...")
    model = RTDETR('rtdetr-l.pt')
    
    print("Initializing lane violation detector...")
    lane_detector = LaneViolationDetector(
        num_lanes=3,
        sequence_length=10,
        deviation_threshold=50.0
    )
    
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
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        timestamp = frame_number / fps
        
        # Detect lane lines
        lane_img, lane_boundaries = detect_lane_lines(frame)
        if lane_boundaries:
            lane_detector.set_lane_boundaries(lane_boundaries)
        
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
        
        # Update lane violation detector
        violations = lane_detector.update(detections, frame_number, timestamp)
        all_violations.extend(violations)
        
        # Visualize
        output_frame = lane_detector.visualize(frame, lane_img, violations)
        
        # Write frame
        out.write(output_frame)
        
        # Progress
        if frame_number % 30 == 0:
            progress = (frame_number / total_frames) * 100
            print(f"Progress: {progress:.1f}% - Frame {frame_number}/{total_frames} - Violations: {len(all_violations)}")
        
        frame_number += 1
    
    cap.release()
    out.release()
    
    # Summary
    print("\n" + "="*60)
    print("Lane Violation Detection Summary")
    print("="*60)
    print(f"Total frames processed: {frame_number}")
    print(f"Total violations detected: {len(all_violations)}")
    
    # Count types
    lane_change_count = sum(1 for v in all_violations if v['type'] == 'IMPROPER_LANE_CHANGE')
    deviation_count = sum(1 for v in all_violations if v['type'] == 'LANE_DEVIATION')
    
    print(f"  - Improper lane changes: {lane_change_count}")
    print(f"  - Lane deviations: {deviation_count}")
    
    print(f"\nOutput saved to: {output_path}")
    print("="*60)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python demo_lane_violation.py <video_path> [output_path]")
        print("Example: python demo_lane_violation.py traffic.mp4 output/lane_result.mp4")
        sys.exit(1)
    
    video_path = sys.argv[1]
    output_path = sys.argv[2] if len(sys.argv) > 2 else "output/lane_violations.mp4"
    
    run_lane_violation_detection(video_path, output_path)
