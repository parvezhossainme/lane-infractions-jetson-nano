"""
Vehicle Speed Tracking and Detection
Detects improper speeds (too slow or too fast) using object tracking.
"""

import cv2
import numpy as np
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional


@dataclass
class VehicleTrack:
    """Store tracking information for a vehicle."""
    id: int
    positions: List[Tuple[float, float]]  # (x, y) center positions
    timestamps: List[float]  # Frame timestamps
    speeds: List[float]  # Calculated speeds
    class_id: int  # Vehicle class (car, truck, etc.)
    

class SpeedDetector:
    """
    Detect vehicles driving too fast or too slow.
    Uses optical flow and tracking to estimate speed.
    """
    
    def __init__(self, 
                 pixels_per_meter: float = 10.0,
                 fps: float = 30.0,
                 min_speed_kmh: float = 20.0,
                 max_speed_kmh: float = 80.0,
                 speed_zone: str = "urban"):
        """
        Initialize speed detector.
        
        Args:
            pixels_per_meter: Calibration - pixels per meter in scene
            fps: Video frame rate
            min_speed_kmh: Minimum allowed speed (km/h)
            max_speed_kmh: Maximum allowed speed (km/h)
            speed_zone: Type of zone (urban, highway, school)
        """
        self.pixels_per_meter = pixels_per_meter
        self.fps = fps
        self.min_speed_kmh = min_speed_kmh
        self.max_speed_kmh = max_speed_kmh
        self.speed_zone = speed_zone
        
        # Track vehicles
        self.tracks: Dict[int, VehicleTrack] = {}
        self.next_id = 0
        
        # Violation tracking
        self.violations = []
        
    def calculate_speed(self, positions: List[Tuple[float, float]], 
                       timestamps: List[float]) -> float:
        """
        Calculate speed from position history.
        
        Args:
            positions: List of (x, y) positions
            timestamps: Corresponding timestamps
            
        Returns:
            Speed in km/h
        """
        if len(positions) < 2:
            return 0.0
        
        # Calculate distance in pixels
        distances = []
        for i in range(1, len(positions)):
            x1, y1 = positions[i-1]
            x2, y2 = positions[i]
            dist = np.sqrt((x2-x1)**2 + (y2-y1)**2)
            distances.append(dist)
        
        # Average distance per frame
        avg_dist_pixels = np.mean(distances)
        
        # Convert to meters
        dist_meters = avg_dist_pixels / self.pixels_per_meter
        
        # Time between frames
        time_seconds = 1.0 / self.fps
        
        # Speed in m/s then convert to km/h
        speed_ms = dist_meters / time_seconds
        speed_kmh = speed_ms * 3.6
        
        return speed_kmh
    
    def match_detection_to_track(self, center: Tuple[float, float], 
                                 max_distance: float = 50.0) -> Optional[int]:
        """
        Match a detection to existing track.
        
        Args:
            center: (x, y) center of detection
            max_distance: Maximum distance for matching
            
        Returns:
            Track ID if matched, None otherwise
        """
        min_dist = float('inf')
        matched_id = None
        
        for track_id, track in self.tracks.items():
            if len(track.positions) > 0:
                last_pos = track.positions[-1]
                dist = np.sqrt((center[0]-last_pos[0])**2 + (center[1]-last_pos[1])**2)
                if dist < min_dist and dist < max_distance:
                    min_dist = dist
                    matched_id = track_id
        
        return matched_id
    
    def update(self, detections: List[Dict], frame_number: int) -> List[Dict]:
        """
        Update tracks with new detections and detect violations.
        
        Args:
            detections: List of detection dicts with 'bbox', 'class_id', 'confidence'
            frame_number: Current frame number
            
        Returns:
            List of speed violations
        """
        timestamp = frame_number / self.fps
        violations = []
        
        # Match detections to tracks
        matched_tracks = set()
        
        for det in detections:
            x1, y1, x2, y2 = det['bbox']
            center_x = (x1 + x2) / 2
            center_y = (y1 + y2) / 2
            center = (center_x, center_y)
            
            # Try to match to existing track
            track_id = self.match_detection_to_track(center)
            
            if track_id is not None:
                # Update existing track
                track = self.tracks[track_id]
                track.positions.append(center)
                track.timestamps.append(timestamp)
                matched_tracks.add(track_id)
                
                # Keep only recent history (last 30 frames)
                if len(track.positions) > 30:
                    track.positions = track.positions[-30:]
                    track.timestamps = track.timestamps[-30:]
                
                # Calculate speed if we have enough data
                if len(track.positions) >= 5:
                    speed = self.calculate_speed(track.positions, track.timestamps)
                    track.speeds.append(speed)
                    
                    # Check for violations
                    if speed > self.max_speed_kmh:
                        violations.append({
                            'type': 'SPEEDING',
                            'vehicle_id': track_id,
                            'speed': speed,
                            'limit': self.max_speed_kmh,
                            'position': center,
                            'frame': frame_number,
                            'severity': 'HIGH' if speed > self.max_speed_kmh * 1.2 else 'MEDIUM'
                        })
                    elif speed < self.min_speed_kmh and speed > 5.0:  # Not stopped
                        violations.append({
                            'type': 'TOO_SLOW',
                            'vehicle_id': track_id,
                            'speed': speed,
                            'limit': self.min_speed_kmh,
                            'position': center,
                            'frame': frame_number,
                            'severity': 'LOW'
                        })
            else:
                # Create new track
                new_track = VehicleTrack(
                    id=self.next_id,
                    positions=[center],
                    timestamps=[timestamp],
                    speeds=[],
                    class_id=det.get('class_id', 0)
                )
                self.tracks[self.next_id] = new_track
                matched_tracks.add(self.next_id)
                self.next_id += 1
        
        # Remove old tracks (not matched for several frames)
        tracks_to_remove = []
        for track_id in self.tracks:
            if track_id not in matched_tracks:
                track = self.tracks[track_id]
                if len(track.timestamps) > 0:
                    time_since_update = timestamp - track.timestamps[-1]
                    if time_since_update > 1.0:  # 1 second
                        tracks_to_remove.append(track_id)
        
        for track_id in tracks_to_remove:
            del self.tracks[track_id]
        
        return violations
    
    def get_track_info(self, track_id: int) -> Optional[Dict]:
        """Get current information about a track."""
        if track_id not in self.tracks:
            return None
        
        track = self.tracks[track_id]
        current_speed = track.speeds[-1] if track.speeds else 0.0
        
        return {
            'id': track_id,
            'position': track.positions[-1] if track.positions else (0, 0),
            'speed': current_speed,
            'avg_speed': np.mean(track.speeds) if track.speeds else 0.0,
            'class_id': track.class_id
        }
    
    def visualize(self, frame: np.ndarray, violations: List[Dict]) -> np.ndarray:
        """
        Draw speed information and violations on frame.
        
        Args:
            frame: Input frame
            violations: List of violations from update()
            
        Returns:
            Annotated frame
        """
        output = frame.copy()
        
        # Draw all tracks
        for track_id, track in self.tracks.items():
            if len(track.positions) > 0:
                x, y = track.positions[-1]
                x, y = int(x), int(y)
                
                # Get speed
                speed = track.speeds[-1] if track.speeds else 0.0
                
                # Color based on speed
                if speed > self.max_speed_kmh:
                    color = (0, 0, 255)  # Red - too fast
                elif speed < self.min_speed_kmh and speed > 5.0:
                    color = (255, 165, 0)  # Orange - too slow
                else:
                    color = (0, 255, 0)  # Green - OK
                
                # Draw speed
                cv2.circle(output, (x, y), 5, color, -1)
                cv2.putText(output, f"{speed:.1f} km/h", (x+10, y),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        # Draw violations
        for violation in violations:
            x, y = violation['position']
            x, y = int(x), int(y)
            
            if violation['type'] == 'SPEEDING':
                text = f"SPEEDING! {violation['speed']:.1f} km/h"
                color = (0, 0, 255)
            else:
                text = f"TOO SLOW! {violation['speed']:.1f} km/h"
                color = (255, 165, 0)
            
            # Draw warning box
            cv2.rectangle(output, (x-50, y-30), (x+150, y+10), color, 2)
            cv2.putText(output, text, (x-45, y-10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 2)
        
        # Draw speed limits
        cv2.putText(output, f"Speed Limit: {self.max_speed_kmh:.0f} km/h", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(output, f"Min Speed: {self.min_speed_kmh:.0f} km/h", (10, 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        return output
