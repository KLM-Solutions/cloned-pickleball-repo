"""
MediaPipe Video Annotator

Processes video frames with MediaPipe Pose detection and draws skeleton overlays.
"""

import cv2
import numpy as np
import mediapipe as mp
from typing import List, Dict, Any, Tuple
from pathlib import Path


class VideoAnnotator:
    """Processes videos with MediaPipe pose detection and skeleton drawing."""
    
    # MediaPipe pose connections for skeleton drawing
    POSE_CONNECTIONS = [
        (0, 1), (1, 2), (2, 3), (3, 7),      # Face
        (0, 4), (4, 5), (5, 6), (6, 8),      # Face
        (9, 10),                              # Mouth
        (11, 12),                             # Shoulders
        (11, 13), (13, 15),                   # Left arm
        (12, 14), (14, 16),                   # Right arm
        (11, 23), (12, 24),                   # Torso
        (23, 24),                             # Hips
        (23, 25), (25, 27),                   # Left leg
        (24, 26), (26, 28),                   # Right leg
        (15, 17), (15, 19), (15, 21),         # Left hand
        (16, 18), (16, 20), (16, 22),         # Right hand
        (27, 29), (27, 31),                   # Left foot
        (28, 30), (28, 32),                   # Right foot
    ]
    
    def __init__(
        self,
        model_complexity: int = 1,
        min_detection_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5
    ):
        """Initialize MediaPipe Pose."""
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=model_complexity,
            enable_segmentation=False,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence
        )
    
    def process_video(
        self,
        input_path: str,
        output_path: str
    ) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        """
        Process video with MediaPipe pose detection.
        
        Args:
            input_path: Path to input video file
            output_path: Path to save annotated video
            
        Returns:
            Tuple of (landmarks_list, metadata)
        """
        cap = cv2.VideoCapture(input_path)
        
        if not cap.isOpened():
            raise ValueError(f"Failed to open video: {input_path}")
        
        # Get video properties
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Setup video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        landmarks_list = []
        frame_index = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Convert BGR to RGB for MediaPipe
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.pose.process(rgb_frame)
            
            # Extract landmarks
            frame_landmarks = None
            if results.pose_landmarks:
                frame_landmarks = self._extract_landmarks(results.pose_landmarks)
                # Draw skeleton on frame
                self._draw_skeleton(frame, results.pose_landmarks, width, height)
            
            # Store landmarks data
            landmarks_list.append({
                "frame_index": frame_index,
                "timestamp_sec": round(frame_index / fps, 3),
                "pose": frame_landmarks
            })
            
            # Write annotated frame
            out.write(frame)
            frame_index += 1
        
        cap.release()
        out.release()
        
        metadata = {
            "fps": int(fps),
            "total_frames": total_frames,
            "duration_sec": round(total_frames / fps, 2),
            "width": width,
            "height": height
        }
        
        return landmarks_list, metadata
    
    def _extract_landmarks(self, pose_landmarks) -> List[Dict[str, float]]:
        """Extract 33 pose landmarks to list of dicts."""
        landmarks = []
        for lm in pose_landmarks.landmark:
            landmarks.append({
                "x": round(lm.x, 4),
                "y": round(lm.y, 4),
                "z": round(lm.z, 4),
                "visibility": round(lm.visibility, 3)
            })
        return landmarks
    
    def _draw_skeleton(
        self,
        frame: np.ndarray,
        pose_landmarks,
        width: int,
        height: int
    ) -> None:
        """Draw skeleton overlay on frame."""
        landmarks = pose_landmarks.landmark
        
        # Draw connections (lines)
        for start_idx, end_idx in self.POSE_CONNECTIONS:
            start = landmarks[start_idx]
            end = landmarks[end_idx]
            
            # Skip if low visibility
            if start.visibility < 0.5 or end.visibility < 0.5:
                continue
            
            start_point = (int(start.x * width), int(start.y * height))
            end_point = (int(end.x * width), int(end.y * height))
            
            cv2.line(frame, start_point, end_point, (0, 255, 0), 2)
        
        # Draw landmarks (circles)
        for lm in landmarks:
            if lm.visibility < 0.5:
                continue
            
            x = int(lm.x * width)
            y = int(lm.y * height)
            
            # Color based on visibility
            if lm.visibility > 0.8:
                color = (0, 255, 0)    # Green - high
            elif lm.visibility > 0.5:
                color = (0, 255, 255)  # Yellow - medium
            else:
                color = (0, 0, 255)    # Red - low
            
            cv2.circle(frame, (x, y), 5, color, -1)
            cv2.circle(frame, (x, y), 5, (255, 255, 255), 1)
    
    def close(self):
        """Release MediaPipe resources."""
        self.pose.close()
