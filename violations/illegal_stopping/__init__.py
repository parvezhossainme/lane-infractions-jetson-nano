"""
Illegal Stopping Detection
Detects vehicles stopping in prohibited zones.
"""

from .illegal_stopping_detector import IllegalStoppingDetector, ZoneType, StoppingZone, VehicleStop

__all__ = ['IllegalStoppingDetector', 'ZoneType', 'StoppingZone', 'VehicleStop']
