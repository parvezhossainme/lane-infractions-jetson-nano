"""
Illegal Stopping Detection
Detects vehicles stopping in prohibited areas using temporal analysis.
"""

import cv2
import numpy as np
from collections import defaultdict
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional, Set
from enum import Enum


class ZoneType(Enum):
    """Types of zones for stopping regulations."""
    NO_STOPPING = "no_stopping"  # Absolutely no stopping
    NO_PARKING = "no_parking"    # Can stop briefly (<3 min)
    LOADING_ZONE = "loading_zone"  # Only for loading/unloading
    BUS_STOP = "bus_stop"        # Only for buses
    ALLOWED = "allowed"          # Parking allowed


@dataclass
class StoppingZone:
    """Define a zone with stopping regulations."""
    x1: float
    y1: float
    x2: float
    y2: float
    zone_type: ZoneType
    max_duration: float  # seconds (0 for no stopping)
    
    def contains_point(self, x: float, y: float) -> bool:
        """Check if point is inside zone."""
        return self.x1 <= x <= self.x2 and self.y1 <= y <= self.y2


@dataclass
class VehicleStop:
    """Track stopped vehicle information."""
    id: int
    position: Tuple[float, float]
    stop_start_time: float
    last_seen_time: float
    zone: Optional[StoppingZone]
    class_id: int
    is_moving: bool
    movement_history: List[float]  # Recent movement amounts


class IllegalStoppingDetector:
    """
    Detect vehicles illegally stopping or parking.
    Uses temporal analysis and zone-based rules.
    """
    
    def __init__(self,
                 movement_threshold: float = 5.0,
                 stationary_time: float = 3.0):
        """
        Initialize illegal stopping detector.
        
        Args:
            movement_threshold: Pixels moved to consider as movement
            stationary_time: Seconds to be considered stopped
        """
        self.movement_threshold = movement_threshold
        self.stationary_time = stationary_time
        
        # Tracking
        self.vehicles: Dict[int, VehicleStop] = {}
        self.next_id = 0
        
        # Zones
        self.zones: List[StoppingZone] = []
        
        # Violations
        self.violations: List[Dict] = []
        
    def add_zone(self, x1: float, y1: float, x2: float, y2: float,
                 zone_type: ZoneType, max_duration: float = 0.0):
        """
        Add a stopping regulation zone.
        
        Args:
            x1, y1, x2, y2: Zone coordinates
            zone_type: Type of zone regulation
            max_duration: Maximum allowed stopping duration (seconds)
        """
        zone = StoppingZone(x1, y1, x2, y2, zone_type, max_duration)
        self.zones.append(zone)
        
    def get_zone_at_position(self, x: float, y: float) -> Optional[StoppingZone]:
        """Get zone at given position."""
        for zone in self.zones:
            if zone.contains_point(x, y):
                return zone
        return None
    
    def calculate_movement(self, prev_pos: Tuple[float, float],
                          curr_pos: Tuple[float, float]) -> float:
        """Calculate movement distance between positions."""
        return np.sqrt((curr_pos[0]-prev_pos[0])**2 + (curr_pos[1]-prev_pos[1])**2)
    
    def is_vehicle_stationary(self, vehicle: VehicleStop) -> bool:
        """
        Determine if vehicle is stationary based on movement history.
        
        Args:
            vehicle: Vehicle stop info
            
        Returns:
            True if stationary
        """
        if len(vehicle.movement_history) < 3:
            return False
        
        # Check if average recent movement is below threshold
        avg_movement = np.mean(vehicle.movement_history[-5:])
        return avg_movement < self.movement_threshold
    
    def match_detection(self, center: Tuple[float, float],
                       max_distance: float = 50.0) -> Optional[int]:
        """Match detection to existing vehicle."""
        min_dist = float('inf')
        matched_id = None
        
        for veh_id, vehicle in self.vehicles.items():
            dist = self.calculate_movement(vehicle.position, center)
            if dist < min_dist and dist < max_distance:
                min_dist = dist
                matched_id = veh_id
        
        return matched_id
    
    def check_violation(self, vehicle: VehicleStop, current_time: float) -> Optional[Dict]:
        """
        Check if a stopped vehicle is violating regulations.
        
        Args:
            vehicle: Vehicle to check
            current_time: Current timestamp
            
        Returns:
            Violation dict if found, None otherwise
        """
        # Check if vehicle is stationary
        if not self.is_vehicle_stationary(vehicle):
            return None
        
        # Calculate how long stopped
        stop_duration = current_time - vehicle.stop_start_time
        
        # Must be stopped for minimum time to count
        if stop_duration < self.stationary_time:
            return None
        
        # Check zone regulations
        zone = vehicle.zone
        
        if zone is None:
            # No zone defined - check generic rules
            # If stopped > 5 minutes in traffic, might be suspicious
            if stop_duration > 300:  # 5 minutes
                return {
                    'type': 'SUSPICIOUS_STOPPING',
                    'duration': stop_duration,
                    'severity': 'LOW'
                }
            return None
        
        # Zone-specific checks
        if zone.zone_type == ZoneType.NO_STOPPING:
            # Any stopping is violation
            return {
                'type': 'NO_STOPPING_ZONE_VIOLATION',
                'zone_type': zone.zone_type.value,
                'duration': stop_duration,
                'severity': 'HIGH'
            }
        
        elif zone.zone_type == ZoneType.NO_PARKING:
            # Brief stops OK, parking not OK
            if stop_duration > zone.max_duration:
                return {
                    'type': 'NO_PARKING_VIOLATION',
                    'zone_type': zone.zone_type.value,
                    'duration': stop_duration,
                    'allowed_duration': zone.max_duration,
                    'severity': 'MEDIUM'
                }
        
        elif zone.zone_type == ZoneType.BUS_STOP:
            # Only buses allowed
            if vehicle.class_id != 5:  # 5 = bus in COCO
                return {
                    'type': 'BUS_STOP_VIOLATION',
                    'zone_type': zone.zone_type.value,
                    'duration': stop_duration,
                    'severity': 'MEDIUM'
                }
        
        elif zone.zone_type == ZoneType.LOADING_ZONE:
            # Time-limited
            if stop_duration > zone.max_duration:
                return {
                    'type': 'LOADING_ZONE_VIOLATION',
                    'zone_type': zone.zone_type.value,
                    'duration': stop_duration,
                    'allowed_duration': zone.max_duration,
                    'severity': 'LOW'
                }
        
        return None
    
    def update(self, detections: List[Dict], timestamp: float) -> List[Dict]:
        """
        Update vehicle tracking and detect violations.
        
        Args:
            detections: List of vehicle detections
            timestamp: Current timestamp
            
        Returns:
            List of violations detected
        """
        current_violations = []
        matched_ids = set()
        
        for det in detections:
            x1, y1, x2, y2 = det['bbox']
            center = ((x1+x2)/2, (y1+y2)/2)
            class_id = det['class_id']
            
            # Match to existing vehicle
            veh_id = self.match_detection(center)
            
            if veh_id is not None:
                # Update existing vehicle
                vehicle = self.vehicles[veh_id]
                
                # Calculate movement
                movement = self.calculate_movement(vehicle.position, center)
                vehicle.movement_history.append(movement)
                
                # Keep only recent history
                if len(vehicle.movement_history) > 10:
                    vehicle.movement_history = vehicle.movement_history[-10:]
                
                # Update position and time
                prev_position = vehicle.position
                vehicle.position = center
                vehicle.last_seen_time = timestamp
                
                # Check if vehicle started or stopped moving
                was_stationary = not vehicle.is_moving
                vehicle.is_moving = movement > self.movement_threshold
                
                # If vehicle stopped moving, record stop time
                if vehicle.is_moving:
                    vehicle.stop_start_time = timestamp
                elif not was_stationary and not vehicle.is_moving:
                    vehicle.stop_start_time = timestamp
                
                # Update zone
                vehicle.zone = self.get_zone_at_position(center[0], center[1])
                
                matched_ids.add(veh_id)
                
                # Check for violation
                violation = self.check_violation(vehicle, timestamp)
                if violation:
                    violation['vehicle_id'] = veh_id
                    violation['position'] = center
                    violation['timestamp'] = timestamp
                    current_violations.append(violation)
                    
            else:
                # New vehicle
                zone = self.get_zone_at_position(center[0], center[1])
                new_vehicle = VehicleStop(
                    id=self.next_id,
                    position=center,
                    stop_start_time=timestamp,
                    last_seen_time=timestamp,
                    zone=zone,
                    class_id=class_id,
                    is_moving=True,
                    movement_history=[0.0]
                )
                self.vehicles[self.next_id] = new_vehicle
                matched_ids.add(self.next_id)
                self.next_id += 1
        
        # Remove vehicles not seen recently
        to_remove = []
        for veh_id in self.vehicles:
            if veh_id not in matched_ids:
                vehicle = self.vehicles[veh_id]
                time_since_seen = timestamp - vehicle.last_seen_time
                if time_since_seen > 5.0:  # 5 seconds
                    to_remove.append(veh_id)
        
        for veh_id in to_remove:
            del self.vehicles[veh_id]
        
        return current_violations
    
    def visualize(self, frame: np.ndarray, violations: List[Dict] = None) -> np.ndarray:
        """
        Visualize zones, stopped vehicles, and violations.
        
        Args:
            frame: Input frame
            violations: List of violations
            
        Returns:
            Annotated frame
        """
        output = frame.copy()
        
        # Draw zones
        for zone in self.zones:
            x1, y1, x2, y2 = int(zone.x1), int(zone.y1), int(zone.x2), int(zone.y2)
            
            # Color based on zone type
            if zone.zone_type == ZoneType.NO_STOPPING:
                color = (0, 0, 255)  # Red
                label = "NO STOPPING"
            elif zone.zone_type == ZoneType.NO_PARKING:
                color = (0, 165, 255)  # Orange
                label = "NO PARKING"
            elif zone.zone_type == ZoneType.BUS_STOP:
                color = (255, 255, 0)  # Cyan
                label = "BUS STOP"
            elif zone.zone_type == ZoneType.LOADING_ZONE:
                color = (0, 255, 255)  # Yellow
                label = "LOADING"
            else:
                color = (0, 255, 0)  # Green
                label = "PARKING OK"
            
            # Draw semi-transparent zone
            overlay = output.copy()
            cv2.rectangle(overlay, (x1, y1), (x2, y2), color, -1)
            output = cv2.addWeighted(output, 0.9, overlay, 0.1, 0)
            
            # Draw border and label
            cv2.rectangle(output, (x1, y1), (x2, y2), color, 2)
            cv2.putText(output, label, (x1+5, y1+20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        # Draw stopped vehicles
        for veh_id, vehicle in self.vehicles.items():
            if self.is_vehicle_stationary(vehicle):
                x, y = int(vehicle.position[0]), int(vehicle.position[1])
                
                # Calculate stop duration
                stop_duration = vehicle.last_seen_time - vehicle.stop_start_time
                
                # Color: red if violation likely, yellow otherwise
                color = (0, 255, 255) if stop_duration > self.stationary_time else (0, 255, 0)
                
                cv2.circle(output, (x, y), 8, color, -1)
                cv2.putText(output, f"STOPPED {stop_duration:.1f}s", (x+15, y),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        # Draw violations
        if violations:
            for violation in violations:
                x, y = int(violation['position'][0]), int(violation['position'][1])
                
                vtype = violation['type']
                if 'NO_STOPPING' in vtype:
                    text = "ILLEGAL STOP!"
                    color = (0, 0, 255)
                elif 'PARKING' in vtype:
                    text = "PARKING VIOLATION!"
                    color = (0, 165, 255)
                elif 'BUS_STOP' in vtype:
                    text = "BUS STOP VIOLATION!"
                    color = (255, 255, 0)
                else:
                    text = "VIOLATION!"
                    color = (255, 0, 0)
                
                # Draw warning
                cv2.rectangle(output, (x-70, y-40), (x+150, y-5), color, -1)
                cv2.putText(output, text, (x-65, y-15),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                
                # Draw duration
                if 'duration' in violation:
                    dur_text = f"{violation['duration']:.0f}s"
                    cv2.putText(output, dur_text, (x-65, y+10),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
        
        return output
