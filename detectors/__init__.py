"""
Core Detection Modules
Contains basic detection functionality for lanes and vehicles.
"""

from .lane_detection import detect_lanes, preprocess_lane, fixed_roi_for_this_image
from .vehicle_detection_ultra import detect_vehicles

__all__ = [
    'detect_lanes',
    'preprocess_lane', 
    'fixed_roi_for_this_image',
    'detect_vehicles'
]
