#!/usr/bin/env python3
"""
Vehicle Detection using Ultralytics RT-DETR
Simpler alternative that works out of the box.
"""

import sys
import os
import cv2
import numpy as np
from ultralytics import RTDETR

# COCO vehicle class IDs for YOLO/RT-DETR
VEHICLE_CLASSES = [2, 3, 5, 7]  # car, motorcycle, bus, truck
CLASS_NAMES = {2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}


def detect_vehicles(image_path, output_dir="output", confidence=0.5):
    """
    Detect vehicles in an image using RT-DETR.
    
    Args:
        image_path: Path to input image
        output_dir: Directory to save results
        confidence: Confidence threshold (0-1)
    """
    # Check if image exists
    if not os.path.exists(image_path):
        print(f"❌ Error: Image not found: {image_path}")
        return False
    
    # Load image
    print(f"Loading image: {image_path}")
    img = cv2.imread(image_path)
    if img is None:
        print(f"❌ Error: Could not read image")
        return False
    
    print(f"✅ Image loaded: {img.shape}")
    
    # Load RT-DETR model (auto-downloads on first run)
    print("\nLoading RT-DETR model...")
    print("(First run will download the model ~45MB)")
    model = RTDETR('rtdetr-l.pt')  # RT-DETR Large model
    print("✅ Model loaded")
    
    # Run detection
    print(f"\nRunning detection (confidence >= {confidence})...")
    results = model(image_path, conf=confidence, verbose=False)
    
    # Process results
    result = results[0]
    boxes = result.boxes
    
    # Filter for vehicles only
    vehicle_detections = []
    for box in boxes:
        cls_id = int(box.cls[0])
        if cls_id in VEHICLE_CLASSES:
            vehicle_detections.append({
                'class_id': cls_id,
                'class_name': CLASS_NAMES.get(cls_id, f'class_{cls_id}'),
                'confidence': float(box.conf[0]),
                'bbox': box.xyxy[0].cpu().numpy()
            })
    
    print(f"\n✅ Detected {len(vehicle_detections)} vehicles:")
    for i, det in enumerate(vehicle_detections, 1):
        print(f"  {i}. {det['class_name']}: {det['confidence']:.2f}")
    
    # Visualize and save
    os.makedirs(output_dir, exist_ok=True)
    
    # Draw detections on image
    output_img = img.copy()
    for det in vehicle_detections:
        x1, y1, x2, y2 = det['bbox'].astype(int)
        
        # Draw box
        cv2.rectangle(output_img, (x1, y1), (x2, y2), (0, 255, 0), 3)
        
        # Draw label with background
        label = f"{det['class_name']} {det['confidence']:.2f}"
        (text_w, text_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
        cv2.rectangle(output_img, (x1, y1-text_h-10), (x1+text_w, y1), (0, 255, 0), -1)
        cv2.putText(output_img, label, (x1, y1-5),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
    
    # Save result
    output_path = os.path.join(output_dir, 'vehicle_detection.jpg')
    cv2.imwrite(output_path, output_img)
    print(f"\n✅ Result saved to: {output_path}")
    
    # Also save the ultralytics annotated version
    result.save(filename=os.path.join(output_dir, 'vehicle_detection_full.jpg'))
    print(f"✅ Full detection saved to: {output_dir}/vehicle_detection_full.jpg")
    
    return True


def main():
    """Main function."""
    if len(sys.argv) < 2:
        print("Usage: python vehicle_detection_ultra.py <image_path> [confidence]")
        print("Example: python vehicle_detection_ultra.py IMG_20250813_161947.jpg 0.5")
        print("\nNote: First run will download RT-DETR model (~45MB)")
        sys.exit(1)
    
    image_path = sys.argv[1]
    confidence = float(sys.argv[2]) if len(sys.argv) > 2 else 0.5
    
    print("="*60)
    print("Vehicle Detection using RT-DETR")
    print("="*60)
    
    success = detect_vehicles(image_path, confidence=confidence)
    
    if success:
        print("\n" + "="*60)
        print("🎉 Vehicle detection complete!")
        print("="*60)
    else:
        sys.exit(1)


if __name__ == "__main__":
    main()
