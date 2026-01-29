import sys
import os

print(f"Python Executable: {sys.executable}")
print(f"Python Version: {sys.version}")

print("\n--- Testing Imports ---")

# 1. MediaPipe
try:
    import mediapipe as mp
    print(f"MediaPipe loaded: {mp.__version__}")
    if hasattr(mp, 'solutions'):
        print("mp.solutions available")
        import mediapipe.python.solutions.pose as mp_pose
        print("mp.solutions.pose available")
    else:
        print("FAIL: mp.solutions NOT available")
except Exception as e:
    print(f"FAIL: MediaPipe import error: {e}")

# 2. Key Custom Modules
try:
    from biomechanics import BiomechanicsAnalyzer
    print("BiomechanicsAnalyzer loaded")
except ImportError as e:
    print(f"FAIL: BiomechanicsAnalyzer import error: {e}")
except Exception as e:
    print(f"FAIL: BiomechanicsAnalyzer other error: {e}")

try:
    from classification import StrokeClassifier
    print("StrokeClassifier loaded")
except ImportError as e:
    print(f"FAIL: StrokeClassifier import error: {e}")
except Exception as e:
    print(f"FAIL: StrokeClassifier other error: {e}")

print("\n--- Done ---")
