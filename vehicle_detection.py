"""
Vehicle Detection Module using RT-DETR
This module contains functions for detecting vehicles in images using RT-DETR model.

IMPORTANT: This module requires RT-DETR to be set up first.
Run: python setup_rtdetr.py
Then run this script from: RT-DETR/rtdetr_pytorch/
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
import sys
import os
from pathlib import Path


# COCO vehicle class IDs
VEHICLE_IDS = [2, 3, 5, 7]  # car, motorcycle, bus, truck


def check_rtdetr_setup():
    """Check if RT-DETR is properly set up."""
    errors = []
    
    # Check if we're in the right directory
    if not os.path.exists('src/core.py'):
        errors.append("Not in RT-DETR directory. Expected to be in RT-DETR/rtdetr_pytorch/")
    
    # Check if torch is available
    try:
        import torch
    except ImportError:
        errors.append("PyTorch not installed. Run: pip install -r requirements.txt")
    
    # Check if RT-DETR modules are available
    try:
        from src.core import YAMLConfig
    except ImportError:
        errors.append("RT-DETR modules not found. Make sure you're in RT-DETR/rtdetr_pytorch/")
    
    # Check if config exists
    if not os.path.exists('configs/rtdetr/rtdetr_r50vd_6x_coco.yml'):
        errors.append("RT-DETR config file not found")
    
    # Check if checkpoint exists
    if not os.path.exists('checkpoints/rtdetr_r50vd_coco.pth'):
        errors.append("Pretrained model not found. Download it first.")
    
    if errors:
        print("[ERROR] RT-DETR Setup Issues:")
        for i, error in enumerate(errors, 1):
            print(f"  {i}. {error}")
        print("\n📋 Setup Instructions:")
        print("  1. Run setup script: python ../../setup_rtdetr.py")
        print("  2. Or manually:")
        print("     - cd to your project root")
        print("     - Run: python setup_rtdetr.py")
        print("  3. Then run this script from RT-DETR/rtdetr_pytorch/:")
        print("     cd RT-DETR/rtdetr_pytorch")
        print("     python ../../vehicle_detection.py <image_path>")
        return False
    
    return True


class VehicleDetector:
    """RT-DETR based vehicle detector."""
    
    def __init__(self, config_path='configs/rtdetr/rtdetr_r50vd_6x_coco.yml',
                 checkpoint_path='checkpoints/rtdetr_r50vd_coco.pth'):
        """
        Initialize the vehicle detector.
        
        Args:
            config_path: Path to RT-DETR config file
            checkpoint_path: Path to pretrained model checkpoint
        """
        import torch
        from src.core import YAMLConfig
        
        self.cfg = YAMLConfig(config_path, resume=checkpoint_path)
        self.model = self.cfg.model.cuda().eval()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Import transforms
        from src.data.transforms import Compose, Resize, ToTensor, Normalize
        
        self.transform = Compose([
            Resize((640, 640)),
            ToTensor(),
            Normalize()
        ])
    
    def preprocess(self, img):
        """
        Preprocess image for RT-DETR inference.
        
        Args:
            img: Input image (RGB format)
            
        Returns:
            input_tensor: Preprocessed tensor ready for inference
        """
        input_tensor, _ = self.transform(img, None)
        input_tensor = input_tensor.unsqueeze(0).to(self.device)
        return input_tensor
    
    def detect(self, img, confidence_threshold=0.5):
        """
        Detect vehicles in the image.
        
        Args:
            img: Input image (RGB format)
            confidence_threshold: Minimum confidence score for detections
            
        Returns:
            boxes: Detected bounding boxes (normalized coordinates)
            labels: Class labels for detected objects
            scores: Confidence scores for detections
        """
        # Preprocess
        input_tensor = self.preprocess(img)
        
        # Inference
        with torch.no_grad():
            outputs = self.model(input_tensor)
        
        # Post-process
        logits = outputs['pred_logits'][0]
        boxes = outputs['pred_boxes'][0]
        
        # Get predictions
        probs = logits.softmax(-1)
        scores, labels = probs.max(-1)
        
        # Filter by confidence
        keep = scores > confidence_threshold
        boxes = boxes[keep]
        labels = labels[keep]
        scores = scores[keep]
        
        # Filter by vehicle classes only
        vehicle_mask = torch.tensor([int(label) in VEHICLE_IDS for label in labels])
        boxes = boxes[vehicle_mask]
        labels = labels[vehicle_mask]
        scores = scores[vehicle_mask]
        
        return boxes, labels, scores
    
    def visualize(self, img, boxes, labels, scores):
        """
        Visualize detected vehicles on the image.
        
        Args:
            img: Original image (RGB)
            boxes: Bounding boxes (normalized coordinates)
            labels: Class labels
            scores: Confidence scores
            
        Returns:
            output: Image with visualized detections
        """
        from src.misc import box_cxcywh_to_xyxy
        
        h, w = img.shape[:2]
        output = img.copy()
        
        for box, label, score in zip(boxes, labels, scores):
            # Convert box format
            x1, y1, x2, y2 = box_cxcywh_to_xyxy(box)
            
            # Scale to image dimensions
            x1 = int(x1 * w)
            x2 = int(x2 * w)
            y1 = int(y1 * h)
            y2 = int(y2 * h)
            
            # Draw bounding box
            cv2.rectangle(output, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # Draw label and score
            label_text = f"{int(label)} {score:.2f}"
            cv2.putText(output, label_text, (x1, y1-5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                       (255, 0, 0), 2)
        
        return output


def main(image_path=None):
    """
    Main function to demonstrate vehicle detection.
    
    Args:
        image_path: Path to the input image (optional)
    """
    # Check RT-DETR setup first
    if not check_rtdetr_setup():
        sys.exit(1)
    
    # Get image path from argument or use default
    if image_path is None:
        if len(sys.argv) > 1:
            image_path = sys.argv[1]
        else:
            print("Usage: python vehicle_detection.py <image_path>")
            print("Example: python vehicle_detection.py ../../IMG_20250813_161947.jpg")
            sys.exit(1)
    
    # Check if file exists
    if not os.path.exists(image_path):
        print(f"Error: Image file not found: {image_path}")
        print(f"Current directory: {os.getcwd()}")
        sys.exit(1)
    
    # Load image
    print(f"Loading image: {image_path}")
    img = cv2.imread(image_path)
    
    if img is None:
        print(f"Error: Could not read image: {image_path}")
        sys.exit(1)
    
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    print(f"Image loaded successfully: {img.shape}")
    
    # Initialize detector
    print("Initializing vehicle detector...")
    detector = VehicleDetector()
    
    # Detect vehicles
    print("Detecting vehicles...")
    boxes, labels, scores = detector.detect(img, confidence_threshold=0.5)
    
    print(f"Detected {len(boxes)} vehicles")
    
    # Visualize results
    output = detector.visualize(img, boxes, labels, scores)
    
    # Display
    plt.figure(figsize=(10, 6))
    plt.imshow(output)
    plt.title("Vehicle Detection using RT-DETR")
    plt.axis('off')
    plt.show()


if __name__ == "__main__":
    main()
