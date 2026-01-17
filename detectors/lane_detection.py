"""
Lane Detection Module
This module contains functions for detecting lanes in images using ROI and Hough transforms.
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
import sys
import os


def fixed_roi_for_this_image(img):
    """
    Create a fixed Region of Interest (ROI) for lane detection.
    
    Args:
        img: Input image
        
    Returns:
        roi: Image with ROI mask applied
        polygon: Polygon coordinates for the ROI
    """
    h, w = img.shape[:2]
    mask = np.zeros_like(img)

    polygon = np.array([[
        (int(0.25*w), int(0.95*h)),  # bottom-left
        (int(0.75*w), int(0.95*h)),  # bottom-right
        (int(0.62*w), int(0.45*h)),  # top-right
        (int(0.38*w), int(0.45*h))   # top-left
    ]], np.int32)

    cv2.fillPoly(mask, polygon, (255,255,255))
    roi = cv2.bitwise_and(img, mask)
    return roi, polygon


def preprocess_lane(img):
    """
    Preprocess image for lane detection by converting to grayscale,
    applying blur, and adaptive thresholding.
    
    Args:
        img: Input image (RGB)
        
    Returns:
        thresh: Binary threshold image
    """
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    # Strong blur to suppress cars
    blur = cv2.GaussianBlur(gray, (7,7), 0)

    # Adaptive threshold works better for dashed lanes
    thresh = cv2.adaptiveThreshold(
        blur, 255,
        cv2.ADAPTIVE_THRESH_MEAN_C,
        cv2.THRESH_BINARY,
        15, -5
    )

    return thresh


def detect_lanes(img, roi_img):
    """
    Detect lanes using Canny edge detection and Hough line transform.
    
    Args:
        img: Original image
        roi_img: ROI-masked image
        
    Returns:
        final: Image with detected lanes overlaid
        lane_img: Image containing only the detected lane lines
    """
    # Preprocess for lane detection
    lane_binary = preprocess_lane(roi_img)
    
    # Edge detection
    edges = cv2.Canny(lane_binary, 50, 150)

    # Hough line transform
    lines = cv2.HoughLinesP(
        edges,
        rho=1,
        theta=np.pi/180,
        threshold=150,
        minLineLength=120,
        maxLineGap=40
    )

    # Create lane overlay image
    lane_img = np.zeros_like(img)

    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]

            # Filter by length
            length = np.sqrt((x2-x1)**2 + (y2-y1)**2)
            if length < 120:
                continue

            # Filter by angle to remove car edges
            angle = abs(np.degrees(np.arctan2(y2-y1, x2-x1)))
            if angle < 60 or angle > 120:
                continue

            cv2.line(lane_img, (x1, y1), (x2, y2), (0, 255, 0), 3)

    # Combine with original image
    final = cv2.addWeighted(img, 0.85, lane_img, 1, 1)
    
    return final, lane_img


def visualize_roi(img, roi_poly):
    """
    Visualize the ROI polygon on the original image.
    
    Args:
        img: Original image
        roi_poly: ROI polygon coordinates
    """
    roi_vis = img.copy()
    cv2.polylines(roi_vis, roi_poly, True, (255, 0, 0), 4)

    plt.figure(figsize=(10, 6))
    plt.imshow(roi_vis)
    plt.title("ROI FOR THIS CAMERA VIEW")
    plt.axis("off")
    plt.show()


def main(image_path=None):
    """
    Main function to demonstrate lane detection.
    
    Args:
        image_path: Path to the input image (optional)
    """
    # Get image path from argument or use default
    if image_path is None:
        if len(sys.argv) > 1:
            image_path = sys.argv[1]
        else:
            print("Usage: python lane_detection.py <image_path>")
            print("Example: python lane_detection.py IMG_20250813_161947.jpg")
            sys.exit(1)
    
    # Check if file exists
    if not os.path.exists(image_path):
        print(f"Error: Image file not found: {image_path}")
        print(f"Current directory: {os.getcwd()}")
        sys.exit(1)
    
    # Load and convert image
    print(f"Loading image: {image_path}")
    img = cv2.imread(image_path)
    
    if img is None:
        print(f"Error: Could not read image: {image_path}")
        sys.exit(1)
    
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    print(f"Image loaded successfully: {img.shape}")

    # Display original image
    plt.figure(figsize=(10, 6))
    plt.imshow(img)
    plt.title("Original Image")
    plt.axis("off")
    plt.show()

    # Apply ROI
    roi_img, roi_poly = fixed_roi_for_this_image(img)
    
    # Visualize ROI
    visualize_roi(img, roi_poly)

    # Detect lanes
    final, lane_img = detect_lanes(img, roi_img)

    # Display result
    plt.figure(figsize=(10, 6))
    plt.imshow(final)
    plt.title("LANE DETECTION – FIXED CAMERA (THIS IMAGE)")
    plt.axis("off")
    plt.show()


if __name__ == "__main__":
    main()
