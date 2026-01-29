"""
MediaPipe Video Processing Module (Tasks API)

Processes video frames with MediaPipe Pose Landmarker (Tasks API) and draws skeleton overlays.
Supports GPU delegation.
"""

import cv2
import json
import os
import subprocess
import numpy as np
import mediapipe as mp
from typing import List, Dict, Any, Tuple

# MediaPipe Tasks API
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
# from mediapipe.framework.formats import landmark_pb2 # Causing import errors

# Helper classes to mimic protobuf for drawing_utils
class NormalizedLandmark:
    def __init__(self, x, y, z, visibility):
        self.x = float(x)
        self.y = float(y)
        self.z = float(z)
        self.visibility = float(visibility)

class NormalizedLandmarkList:
    def __init__(self):
        self.landmark = []

class NumpyEncoder(json.JSONEncoder):
    """Custom encoder for numpy data types"""
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


class VideoAnnotator:
    """Processes videos with MediaPipe Pose Landmarker (Tasks API)."""
    
    # Landmark names for mapping indices to strings
    LANDMARK_NAMES = [
        "nose",
        "left_eye_inner", "left_eye", "left_eye_outer",
        "right_eye_inner", "right_eye", "right_eye_outer",
        "left_ear", "right_ear",
        "mouth_left", "mouth_right",
        "left_shoulder", "right_shoulder",
        "left_elbow", "right_elbow",
        "left_wrist", "right_wrist",
        "left_pinky", "right_pinky",
        "left_index", "right_index",
        "left_thumb", "right_thumb",
        "left_hip", "right_hip",
        "left_knee", "right_knee",
        "left_ankle", "right_ankle",
        "left_heel", "right_heel",
        "left_foot_index", "right_foot_index",
    ]
    
    def __init__(
        self,
        model_asset_path: str = 'models/pose_landmarker_heavy.task',
        min_pose_detection_confidence: float = 0.5,
        min_pose_presence_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5
    ):
        """Initialize MediaPipe Pose Landmarker."""
        
        # Check if running on RunPod (or forced GPU environment)
        # We try to use GPU if available
        base_options = python.BaseOptions(
            model_asset_path=model_asset_path
        )
        
        # Try to use GPU delegate, fallback to CPU if fails (handled by MP usually, or we can explicit check)
        # For now, we set it based on availability or simple try/except logic is harder with Options
        # We will assume GPU availability if the user asked for it, but for local dev on Windows without CUDA mp
        # it might crash.
        # Let's try to detect if we can use GPU.
        try:
            # Simple heuristic or just try setting it. 
            # If we are on RunPod, we definitely want GPU.
            if os.environ.get("RUNPOD_POD_ID") or os.environ.get("CUDA_VISIBLE_DEVICES"):
                print("Configuring MediaPipe for GPU...")
                base_options.delegate = python.BaseOptions.Delegate.GPU
            else:
                print("Configuring MediaPipe for CPU (default)...")
                # base_options.delegate = python.BaseOptions.Delegate.CPU # Default
        except Exception as e:
            print(f"Warning setting GPU delegate strategy: {e}. Using default.")

        options = vision.PoseLandmarkerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.VIDEO,
            min_pose_detection_confidence=min_pose_detection_confidence,
            min_pose_presence_confidence=min_pose_presence_confidence,
            min_tracking_confidence=min_tracking_confidence,
            output_segmentation_masks=False
        )
        
        self.landmarker = vision.PoseLandmarker.create_from_options(options)
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_pose = mp.solutions.pose
    
    def process_video(
        self,
        input_path: str,
        output_path: str
    ) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        """
        Process video with MediaPipe Pose Landmarker.
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
        
        print(f"Processing {total_frames} frames at {fps} FPS using Tasks API...")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Convert BGR to RGB for MediaPipe
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
            
            # Detect pose
            # Timestamp required for VIDEO running mode (in ms)
            timestamp_ms = int((frame_index / fps) * 1000)
            detection_result = self.landmarker.detect_for_video(mp_image, timestamp_ms)
            
            # Extract landmarks with names
            frame_landmarks_list = None
            if detection_result.pose_landmarks:
                # pose_landmarks is a list of lists (one per detected person). We take the first one.
                if len(detection_result.pose_landmarks) > 0:
                    pose_landmarks = detection_result.pose_landmarks[0]
                    frame_landmarks_list = self._extract_landmarks(pose_landmarks)
                    
                    # Draw skeleton
                    self._draw_skeleton(frame, pose_landmarks)
            
            # Store landmarks data matching original format
            landmarks_list.append({
                "frameIdx": frame_index,
                "timestampSec": round(frame_index / fps, 3),
                "landmarks": frame_landmarks_list
            })
            
            # Write annotated frame
            out.write(frame)
            frame_index += 1
            
            # Progress logging
            if frame_index % 100 == 0:
                print(f"  Processed {frame_index}/{total_frames} frames...")
        
        cap.release()
        out.release()
        
        # Convert to H.264
        self._convert_and_cleanup(output_path)
        
        metadata = {
            "fps": int(fps),
            "total_frames": total_frames,
            "duration_sec": round(total_frames / fps, 2),
            "width": width,
            "height": height
        }
        
        return landmarks_list, metadata
    
    def _extract_landmarks(self, pose_landmarks) -> List[Dict[str, Any]]:
        """Extract landmarks from NormalizedLandmark objects."""
        landmarks = []
        for i, lm in enumerate(pose_landmarks):
            name = self.LANDMARK_NAMES[i] if i < len(self.LANDMARK_NAMES) else ""
            landmarks.append({
                "name": name,
                "x": float(lm.x),
                "y": float(lm.y),
                "z": float(lm.z),
                "visibility": float(lm.visibility)
            })
        return landmarks
    
    def _draw_skeleton(
        self,
        frame: np.ndarray,
        pose_landmarks_list
    ) -> None:
        """
        Draw skeleton overlay.
        Args:
            pose_landmarks_list: List of NormalizedLandmark objects (for one person)
        """
        
        # Convert Tasks API landmarks (list) to Solutions API proto format for reuse of drawing_utils
        pose_landmarks_proto = NormalizedLandmarkList()
        pose_landmarks_proto.landmark.extend([
            NormalizedLandmark(x=lm.x, y=lm.y, z=lm.z, visibility=lm.visibility) 
            for lm in pose_landmarks_list
        ])
        
        # Specs
        landmark_spec = self.mp_drawing.DrawingSpec(
            color=(0, 0, 255),     # RED joints
            thickness=4,
            circle_radius=4
        )
        connection_spec = self.mp_drawing.DrawingSpec(
            color=(0, 255, 255),   # YELLOW connections
            thickness=3,
            circle_radius=2
        )
        
        # Filter connections (no head)
        filtered_connections = [
            c for c in self.mp_pose.POSE_CONNECTIONS 
            if c[0] > 10 and c[1] > 10
        ]
        
        # Hide head landmarks
        for idx in range(11):
            if idx < len(pose_landmarks_proto.landmark):
                pose_landmarks_proto.landmark[idx].visibility = 0.0
                
        try:
            self.mp_drawing.draw_landmarks(
                frame,
                pose_landmarks_proto,
                filtered_connections,
                landmark_drawing_spec=landmark_spec,
                connection_drawing_spec=connection_spec
            )
        except Exception as e:
            print(f"Drawing error: {e}")

    def _convert_and_cleanup(self, output_path: str):
        """Helper to convert to h264."""
        print("Converting to H.264...")
        temp_raw = output_path + ".raw.mp4"
        try:
            if os.path.exists(output_path):
                os.rename(output_path, temp_raw)
                self._convert_to_h264(temp_raw, output_path)
                if os.path.exists(temp_raw):
                    os.remove(temp_raw)
        except Exception as e:
            print(f"Warning: H.264 conversion failed: {e}")
            if os.path.exists(temp_raw) and not os.path.exists(output_path):
                os.rename(temp_raw, output_path)

    def _convert_to_h264(self, input_path: str, output_path: str) -> None:
        """FFmpeg conversion."""
        try:
            cmd = [
                'ffmpeg', '-i', input_path, '-c:v', 'libx264',
                '-pix_fmt', 'yuv420p', '-crf', '23', '-preset', 'fast',
                '-y', '-loglevel', 'error', output_path
            ]
            subprocess.run(cmd, check=True)
            print("✓ Converted to H.264")
        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"FFmpeg failed with code {e.returncode}")
    
    def close(self):
        """Release resources."""
        if hasattr(self, 'landmarker'):
            self.landmarker.close()
