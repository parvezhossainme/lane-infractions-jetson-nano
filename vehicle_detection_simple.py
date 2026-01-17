#!/usr/bin/env python3
"""
Simple Vehicle Detection using RT-DETR
Standalone script that works from the project root directory.
"""

import sys
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

# Add RT-DETR to Python path
sys.path.insert(0, 'RT-DETR/rtdetr_pytorch')

import torch
from src.core import YAMLConfig
from src.data.transforms import Compose, Resize, ToTensor, Normalize
from src.misc import box_cxcywh_to_xyxy

# COCO vehicle class IDs
VEHICLE_IDS = [2, 3, 5, 7]  # car, motorcycle, bus, truck
CLASS_NAMES = {2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}


def load_model():
    """Load RT-DETR model."""
    config_path = 'RT-DETR/rtdetr_pytorch/configs/rtdetr/rtdetr_r50vd_6x_coco.yml'
    checkpoint_path = 'RT-DETR/rtdetr_pytorch/checkpoints/rtdetr_r50vd_coco.pth'
    
    print("Loading RT-DETR model...")
    cfg = YAMLConfig(config_path, resume=checkpoint_path)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = cfg.model.to(device).eval()
    
    print(f"✅ Model loaded on {device}")
    return model, device


def preprocess_image(img):
    """Preprocess image for RT-DETR."""
    transform = Compose([
        Resize((640, 640)),
        ToTensor(),
        Normalize()
    ])
    
    input_tensor, _ = transform(img, None)
    return input_tensor.unsqueeze(0)


def detect_vehicles(model, device, img, confidence_threshold=0.5):
    """Detect vehicles in image."""
    # Preprocess
    input_tensor = preprocess_image(img).to(device)
    
    # Inference
    print("Running detection...")
    with torch.no_grad():
        outputs = model(input_tensor)
    
    # Post-process
    logits = outputs['pred_logits'][0]
    boxes = outputs['pred_boxes'][0]
    
    probs = logits.softmax(-1)
    scores, labels = probs.max(-1)
    
    # Filter by confidence
    keep = scores > confidence_threshold
    boxes = boxes[keep]
    labels = labels[keep]
    scores = scores[keep]
    
    # Filter vehicle classes only
    vehicle_mask = torch.tensor([int(label) in VEHICLE_IDS for label in labels])
    boxes = boxes[vehicle_mask]
    labels = labels[vehicle_mask]
    scores = scores[vehicle_mask]
    
    return boxes.cpu(), labels.cpu(), scores.cpu()


def visualize_detections(img, boxes, labels, scores, save_path="output/vehicle_detection.png"):
    """Visualize detected vehicles."""
    h, w = img.shape[:2]
    output = img.copy()
    
    print(f"\n✅ Detected {len(boxes)} vehicles:")
    
    for i, (box, label, score) in enumerate(zip(boxes, labels, scores), 1):
        # Convert box format
        x1, y1, x2, y2 = box_cxcywh_to_xyxy(box)
        
        # Scale to image dimensions
        x1 = int(x1 * w)
        x2 = int(x2 * w)
        y1 = int(y1 * h)
        y2 = int(y2 * h)
        
        # Get class name
        class_id = int(label)
        class_name = CLASS_NAMES.get(class_id, f"class_{class_id}")
        
        print(f"  {i}. {class_name}: {score:.2f} at ({x1}, {y1}, {x2}, {y2})")
        
        # Draw bounding box
        cv2.rectangle(output, (x1, y1), (x2, y2), (0, 255, 0), 3)
        
        # Draw label
        label_text = f"{class_name} {score:.2f}"
        
        # Background for text
        (text_w, text_h), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
        cv2.rectangle(output, (x1, y1-text_h-10), (x1+text_w, y1), (0, 255, 0), -1)
        
        cv2.putText(output, label_text, (x1, y1-5),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                   (0, 0, 0), 2)
    
    # Save result
    os.makedirs("output", exist_ok=True)
    output_bgr = cv2.cvtColor(output, cv2.COLOR_RGB2BGR)
    cv2.imwrite(save_path, output_bgr)
    print(f"\n✅ Result saved to: {save_path}")
    
    return output


def main():
    """Main function."""
    if len(sys.argv) < 2:
        print("Usage: python vehicle_detection_simple.py <image_path>")
        print("Example: python vehicle_detection_simple.py IMG_20250813_161947.jpg")
        sys.exit(1)
    
    image_path = sys.argv[1]
    
    # Check if image exists
    if not os.path.exists(image_path):
        print(f"❌ Error: Image not found: {image_path}")
        sys.exit(1)
    
    # Check if RT-DETR is set up
    if not os.path.exists('RT-DETR/rtdetr_pytorch/checkpoints/rtdetr_r50vd_coco.pth'):
        print("❌ RT-DETR not set up!")
        print("Run: python setup_rtdetr.py")
        sys.exit(1)
    
    # Load image
    print(f"Loading image: {image_path}")
    img = cv2.imread(image_path)
    if img is None:
        print(f"❌ Error: Could not read image: {image_path}")
        sys.exit(1)
    
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    print(f"✅ Image loaded: {img.shape}")
    
    # Load model
    model, device = load_model()
    
    # Detect vehicles
    boxes, labels, scores = detect_vehicles(model, device, img, confidence_threshold=0.5)
    
    # Visualize
    output = visualize_detections(img, boxes, labels, scores)
    
    print("\n🎉 Vehicle detection complete!")


if __name__ == "__main__":
    main()
