import sys
import os

print(f"Python Executable: {sys.executable}")
print(f"Python Version: {sys.version}")

print("\n--- Testing Imports ---")

# 1. MediaPipe Tasks
try:
    import mediapipe as mp
    print(f"✓ MediaPipe loaded: {mp.__version__}")
    
    from mediapipe.tasks import python
    from mediapipe.tasks.python import vision
    print("✓ MediaPipe Tasks API available (python.vision)")
    
    if hasattr(mp, 'solutions'):
        print("  mp.solutions available (for drawing utilities)")
except Exception as e:
    print(f"✗ MediaPipe import error: {e}")

# 2. OpenCV
try:
    import cv2
    print(f"✓ OpenCV loaded: {cv2.__version__}")
except Exception as e:
    print(f"✗ OpenCV import error: {e}")

# 3. NumPy
try:
    import numpy as np
    print(f"✓ NumPy loaded: {np.__version__}")
except Exception as e:
    print(f"✗ NumPy import error: {e}")

# 4. Model File
model_path = 'models/pose_landmarker_heavy.task'
if os.path.exists(model_path):
    size_mb = os.path.getsize(model_path) / (1024 * 1024)
    print(f"✓ Model file found: {model_path} ({size_mb:.2f} MB)")
else:
    print(f"✗ Model file NOT found: {model_path}")

# 5. Track module
try:
    from track import VideoAnnotator
    print("✓ VideoAnnotator loaded (syntax check passed)")
except ImportError as e:
    print(f"✗ VideoAnnotator import error: {e}")
except Exception as e:
    print(f"✗ VideoAnnotator other error: {e}")

print("\n--- Environment ---")
print(f"SUPABASE_URL: {'SET' if os.environ.get('SUPABASE_URL') else 'NOT SET'}")
print(f"RUNPOD_POD_ID: {'SET' if os.environ.get('RUNPOD_POD_ID') else 'NOT SET'}")

print("\n--- Done ---")
