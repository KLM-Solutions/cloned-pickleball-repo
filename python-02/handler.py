"""
RunPod Serverless Handler for MediaPipe Video Analysis

This handler receives video analysis jobs, processes them, and uploads ONLY 
the annotated video to Supabase Storage.

NOTE: All DATABASE writes are handled by frontend, NOT here.
Python only:
1. Downloads video
2. Runs MediaPipe pose analysis
3. Uploads annotated video to Supabase Storage
4. Returns results to frontend (which stores in DB)

Expected input:
{
    "video_url": "https://supabase.../storage/v1/object/public/videos/video.mp4",
    "output_bucket": "annotated-videos",           # Bucket for output video
    "job_id": "uuid-optional"                      # Optional: for reference only
}

Returns:
{
    "status": "success",
    "job_id": "...",
    "annotated_video_url": "https://supabase.../annotated-videos/job123.mp4",
    "landmarks": [
        {"frame_index": 0, "timestamp_sec": 0.033, "pose": [...]}
    ],
    "metadata": {"fps": 30, "total_frames": 150, "duration_sec": 5.0}
}
"""

import runpod
import os
import sys
import json
import tempfile
import shutil
import time
import uuid
from pathlib import Path

import requests

# Add current directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from track import VideoAnnotator, NumpyEncoder
from supabase_client import get_uploader


def download_video(url: str, dest_path: str) -> bool:
    """Download video from URL to local path."""
    try:
        t0 = time.time()
        print(f"[STEP 1/4] Downloading video from: {url}")
        response = requests.get(url, stream=True, timeout=300)
        response.raise_for_status()
        
        with open(dest_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        
        file_size = os.path.getsize(dest_path) / (1024 * 1024)
        print(f"✓ Video downloaded: {dest_path} ({file_size:.1f} MB) in {time.time()-t0:.2f}s")
        return True
    except Exception as e:
        print(f"✗ Failed to download video: {e}")
        return False


def handler(job):
    """
    RunPod serverless handler function.
    """
    start_time = time.time()
    job_input = job.get("input", {})
    
    # Validate required input
    video_url = job_input.get("video_url")
    if not video_url:
        return {"error": "Missing required field: video_url"}
    
    job_id = job_input.get("job_id") or str(uuid.uuid4())
    output_bucket = job_input.get("output_bucket", "annotated-videos")
    
    print(f"\n=== Starting Job {job_id} ===")
    print(f"Video: {video_url}")
    print(f"Output Bucket: {output_bucket}")
    
    uploader = get_uploader()
    work_dir = tempfile.mkdtemp(prefix="mediapipe_analysis_")
    
    try:
        # 1. Download video
        video_path = os.path.join(work_dir, "input.mp4")
        if not download_video(video_url, video_path):
            return {"error": "Failed to download video from URL"}
        
        # 2. Process with MediaPipe
        print(f"[STEP 2/4] Running MediaPipe pose detection...")
        t0 = time.time()
        
        output_video_path = os.path.join(work_dir, "annotated.mp4")
        annotator = VideoAnnotator()
        landmarks, metadata = annotator.process_video(video_path, output_video_path)
        annotator.close()
        
        print(f"✓ Analysis complete: {metadata['total_frames']} frames in {time.time()-t0:.2f}s")
        
        # 3. Upload annotated video
        print(f"[STEP 3/4] Uploading annotated video...")
        annotated_video_url = None
        
        if os.path.exists(output_video_path):
            annotated_video_url = uploader.upload_file(
                bucket=output_bucket,
                file_path=output_video_path,
                destination_path=f"{job_id}.mp4",
                content_type="video/mp4"
            )

        # 4. Upload results JSON (for large data)
        print(f"[STEP 4/5] Uploading results JSON...")
        results_json_url = None
        results_path = os.path.join(work_dir, "results.json")
        
        # Construct results object
        results_data = {
            "job_id": job_id,
            "frames": landmarks,
            "metadata": metadata
        }
        
        # Save to file
        with open(results_path, 'w') as f:
            json.dump(results_data, f, cls=NumpyEncoder)
            
        # Upload to analysis-results bucket
        if os.path.exists(results_path):
            results_json_url = uploader.upload_file(
                bucket="analysis-results",
                file_path=results_path,
                destination_path=f"{job_id}/results.json",
                content_type="application/json"
            )

        # 5. Build response
        print(f"[STEP 5/5] Building response...")
        processing_time = time.time() - start_time
        
        print(f"=== JOB COMPLETE: {job_id} in {processing_time:.1f}s ===")
        
        # Optimizing payload size: if JSON URL exists, don't send frames inline
        # This prevents FUNCTION_PAYLOAD_TOO_LARGE errors in Serverless/Lambda
        response_landmarks = [] if results_json_url else landmarks
        
        return {
            "status": "success",
            "job_id": job_id,
            "annotated_video_url": annotated_video_url,
            "results_json_url": results_json_url,
            "frames": response_landmarks, # Empty if URL exists, otherwise full data
            "landmarks": response_landmarks, # Alias for backward compatibility
            "metadata": {
                **metadata,
                "processing_time_sec": round(processing_time, 2)
            }
        }
    
    except Exception as e:
        import traceback
        traceback.print_exc()
        return {"error": f"Handler exception: {str(e)}"}
    
    finally:
        try:
            shutil.rmtree(work_dir)
            print(f"Cleaned up: {work_dir}")
        except Exception as e:
            print(f"Cleanup failed: {e}")


# Start the RunPod serverless worker
if __name__ == "__main__":
    print("=" * 60)
    print("MediaPipe Video Analysis Worker")
    print("=" * 60)
    print(f"Supabase URL: {os.environ.get('SUPABASE_URL', 'NOT SET')}")
    print(f"Supabase Key: {'SET' if os.environ.get('SUPABASE_SERVICE_KEY') else 'NOT SET'}")
    print("=" * 60)
    
    # Startup diagnostics
    print("\n--- Startup Diagnostics ---")
    print(f"Working directory: {os.getcwd()}")
    print(f"Python executable: {sys.executable}")
    
    # Test imports
    print("\n--- Testing imports ---")
    try:
        import mediapipe as mp
        print(f"✓ mediapipe: {mp.__version__}")
    except Exception as e:
        print(f"✗ mediapipe import failed: {e}")
    
    try:
        import cv2
        print(f"✓ opencv: {cv2.__version__}")
    except Exception as e:
        print(f"✗ opencv import failed: {e}")
    
    print("=" * 60)
    print("Starting RunPod Serverless Worker...")
    print("=" * 60 + "\n")
    
    runpod.serverless.start({"handler": handler})
