"""
Lane Violation Detection with LSTM
Detects vehicles crossing lane lines without proper indication.
Uses LSTM to predict vehicle trajectory and detect anomalous lane changes.
"""

import cv2
import numpy as np
import torch
import torch.nn as nn
from collections import deque
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass


@dataclass
class VehicleTrajectory:
    """Store trajectory information for a vehicle."""
    id: int
    positions: deque  # Recent (x, y) positions
    lane_positions: deque  # Which lane (0, 1, 2, etc.)
    timestamps: deque
    lane_changes: List[Dict]  # History of lane changes
    is_indicating: bool  # Blinker detection (placeholder)


class TrajectoryLSTM(nn.Module):
    """LSTM model to predict vehicle trajectory."""
    
    def __init__(self, input_size=2, hidden_size=64, num_layers=2, output_size=2):
        super(TrajectoryLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
        
    def forward(self, x):
        # x shape: (batch, sequence, features)
        lstm_out, _ = self.lstm(x)
        # Take last output
        predictions = self.fc(lstm_out[:, -1, :])
        return predictions


class LaneViolationDetector:
    """
    Detect lane violations using lane detection and vehicle tracking.
    Uses LSTM to predict normal trajectory and detect anomalous lane changes.
    """
    
    def __init__(self, 
                 num_lanes: int = 3,
                 sequence_length: int = 10,
                 deviation_threshold: float = 50.0):
        """
        Initialize lane violation detector.
        
        Args:
            num_lanes: Number of lanes in the road
            sequence_length: Length of trajectory sequence for LSTM
            deviation_threshold: Pixel threshold for lane change detection
        """
        self.num_lanes = num_lanes
        self.sequence_length = sequence_length
        self.deviation_threshold = deviation_threshold
        
        # Vehicle trajectories
        self.trajectories: Dict[int, VehicleTrajectory] = {}
        self.next_id = 0
        
        # Lane boundaries (will be set from lane detection)
        self.lane_boundaries: List[float] = []
        
        # LSTM model for trajectory prediction
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.lstm_model = TrajectoryLSTM().to(self.device)
        self.lstm_model.eval()  # Inference mode
        
        # Violations
        self.violations = []
        
    def set_lane_boundaries(self, boundaries: List[float]):
        """
        Set lane boundary positions (x-coordinates).
        
        Args:
            boundaries: List of x-coordinates for lane boundaries
        """
        self.lane_boundaries = sorted(boundaries)
        
    def get_lane_from_position(self, x: float) -> int:
        """
        Determine which lane a vehicle is in based on x position.
        
        Args:
            x: X-coordinate of vehicle center
            
        Returns:
            Lane index (0, 1, 2, ...)
        """
        if not self.lane_boundaries:
            return 0
        
        for i, boundary in enumerate(self.lane_boundaries):
            if x < boundary:
                return i
        return len(self.lane_boundaries)
    
    def predict_trajectory(self, positions: List[Tuple[float, float]]) -> Tuple[float, float]:
        """
        Predict next position using LSTM.
        
        Args:
            positions: List of recent (x, y) positions
            
        Returns:
            Predicted (x, y) position
        """
        if len(positions) < 2:
            return positions[-1] if positions else (0, 0)
        
        # Prepare input sequence
        sequence = np.array(positions[-self.sequence_length:])
        
        # Normalize
        mean = sequence.mean(axis=0)
        std = sequence.std(axis=0) + 1e-6
        sequence_norm = (sequence - mean) / std
        
        # Convert to tensor
        x = torch.FloatTensor(sequence_norm).unsqueeze(0).to(self.device)
        
        # Predict
        with torch.no_grad():
            pred_norm = self.lstm_model(x).cpu().numpy()[0]
        
        # Denormalize
        predicted = pred_norm * std + mean
        
        return tuple(predicted)
    
    def detect_lane_change(self, trajectory: VehicleTrajectory) -> Optional[Dict]:
        """
        Detect if vehicle is changing lanes.
        
        Args:
            trajectory: Vehicle trajectory data
            
        Returns:
            Lane change info if detected, None otherwise
        """
        if len(trajectory.lane_positions) < 2:
            return None
        
        # Check if lane changed
        current_lane = trajectory.lane_positions[-1]
        previous_lane = trajectory.lane_positions[-2]
        
        if current_lane != previous_lane:
            # Lane change detected
            current_pos = trajectory.positions[-1]
            
            # Check if it's a violation (no indicator)
            is_violation = not trajectory.is_indicating
            
            return {
                'from_lane': previous_lane,
                'to_lane': current_lane,
                'position': current_pos,
                'is_violation': is_violation,
                'timestamp': trajectory.timestamps[-1]
            }
        
        return None
    
    def check_lane_deviation(self, trajectory: VehicleTrajectory) -> Optional[Dict]:
        """
        Check if vehicle is deviating from expected trajectory using LSTM.
        
        Args:
            trajectory: Vehicle trajectory data
            
        Returns:
            Deviation info if anomalous, None otherwise
        """
        if len(trajectory.positions) < self.sequence_length:
            return None
        
        # Get actual position
        actual_pos = trajectory.positions[-1]
        
        # Predict position
        positions_list = list(trajectory.positions)
        predicted_pos = self.predict_trajectory(positions_list[:-1])
        
        # Calculate deviation
        deviation = np.sqrt((actual_pos[0]-predicted_pos[0])**2 + 
                          (actual_pos[1]-predicted_pos[1])**2)
        
        if deviation > self.deviation_threshold:
            return {
                'type': 'TRAJECTORY_DEVIATION',
                'deviation': deviation,
                'actual': actual_pos,
                'predicted': predicted_pos,
                'timestamp': trajectory.timestamps[-1]
            }
        
        return None
    
    def match_detection(self, center: Tuple[float, float], 
                       max_distance: float = 80.0) -> Optional[int]:
        """Match detection to existing trajectory."""
        min_dist = float('inf')
        matched_id = None
        
        for traj_id, traj in self.trajectories.items():
            if len(traj.positions) > 0:
                last_pos = traj.positions[-1]
                dist = np.sqrt((center[0]-last_pos[0])**2 + (center[1]-last_pos[1])**2)
                if dist < min_dist and dist < max_distance:
                    min_dist = dist
                    matched_id = traj_id
        
        return matched_id
    
    def update(self, detections: List[Dict], frame_number: int, 
               timestamp: float) -> List[Dict]:
        """
        Update trajectories and detect violations.
        
        Args:
            detections: List of vehicle detections
            frame_number: Current frame number
            timestamp: Current timestamp
            
        Returns:
            List of detected violations
        """
        violations = []
        matched_ids = set()
        
        for det in detections:
            x1, y1, x2, y2 = det['bbox']
            center = ((x1+x2)/2, (y1+y2)/2)
            
            # Match to trajectory
            traj_id = self.match_detection(center)
            
            if traj_id is not None:
                # Update existing trajectory
                traj = self.trajectories[traj_id]
                traj.positions.append(center)
                traj.timestamps.append(timestamp)
                
                # Determine lane
                lane = self.get_lane_from_position(center[0])
                traj.lane_positions.append(lane)
                
                matched_ids.add(traj_id)
                
                # Check for lane change
                lane_change = self.detect_lane_change(traj)
                if lane_change and lane_change['is_violation']:
                    violations.append({
                        'type': 'IMPROPER_LANE_CHANGE',
                        'vehicle_id': traj_id,
                        'from_lane': lane_change['from_lane'],
                        'to_lane': lane_change['to_lane'],
                        'position': center,
                        'frame': frame_number,
                        'severity': 'HIGH'
                    })
                    traj.lane_changes.append(lane_change)
                
                # Check for trajectory deviation
                deviation = self.check_lane_deviation(traj)
                if deviation:
                    violations.append({
                        'type': 'LANE_DEVIATION',
                        'vehicle_id': traj_id,
                        'deviation': deviation['deviation'],
                        'position': center,
                        'frame': frame_number,
                        'severity': 'MEDIUM'
                    })
            else:
                # Create new trajectory
                new_traj = VehicleTrajectory(
                    id=self.next_id,
                    positions=deque(maxlen=self.sequence_length),
                    lane_positions=deque(maxlen=self.sequence_length),
                    timestamps=deque(maxlen=self.sequence_length),
                    lane_changes=[],
                    is_indicating=False  # TODO: Implement blinker detection
                )
                new_traj.positions.append(center)
                new_traj.timestamps.append(timestamp)
                new_traj.lane_positions.append(self.get_lane_from_position(center[0]))
                
                self.trajectories[self.next_id] = new_traj
                matched_ids.add(self.next_id)
                self.next_id += 1
        
        # Remove old trajectories
        to_remove = []
        for traj_id in self.trajectories:
            if traj_id not in matched_ids:
                traj = self.trajectories[traj_id]
                if len(traj.timestamps) > 0:
                    time_diff = timestamp - traj.timestamps[-1]
                    if time_diff > 2.0:  # 2 seconds
                        to_remove.append(traj_id)
        
        for traj_id in to_remove:
            del self.trajectories[traj_id]
        
        return violations
    
    def visualize(self, frame: np.ndarray, lane_lines: Optional[np.ndarray] = None,
                  violations: List[Dict] = None) -> np.ndarray:
        """
        Visualize lane boundaries, trajectories, and violations.
        
        Args:
            frame: Input frame
            lane_lines: Lane line image (optional)
            violations: List of violations
            
        Returns:
            Annotated frame
        """
        output = frame.copy()
        
        # Overlay lane lines if provided
        if lane_lines is not None:
            output = cv2.addWeighted(output, 0.7, lane_lines, 0.3, 0)
        
        # Draw lane boundaries
        for boundary in self.lane_boundaries:
            cv2.line(output, (int(boundary), 0), (int(boundary), frame.shape[0]),
                    (255, 255, 0), 2)
        
        # Draw trajectories
        for traj_id, traj in self.trajectories.items():
            if len(traj.positions) > 1:
                # Draw trajectory path
                points = [(int(p[0]), int(p[1])) for p in traj.positions]
                for i in range(len(points)-1):
                    cv2.line(output, points[i], points[i+1], (0, 255, 255), 2)
                
                # Draw current position
                x, y = points[-1]
                cv2.circle(output, (x, y), 5, (0, 255, 0), -1)
                
                # Show lane number
                lane = traj.lane_positions[-1] if traj.lane_positions else 0
                cv2.putText(output, f"Lane {lane}", (x+10, y),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        
        # Draw violations
        if violations:
            for violation in violations:
                x, y = violation['position']
                x, y = int(x), int(y)
                
                if violation['type'] == 'IMPROPER_LANE_CHANGE':
                    text = f"IMPROPER LANE CHANGE"
                    color = (0, 0, 255)
                else:
                    text = f"LANE DEVIATION"
                    color = (255, 165, 0)
                
                cv2.rectangle(output, (x-60, y-30), (x+180, y+10), color, 2)
                cv2.putText(output, text, (x-55, y-10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        return output
