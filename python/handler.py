"""
RunPod Serverless Handler for Pickleball Video Analysis

This handler receives video analysis jobs, processes them, and uploads ONLY 
the annotated video to Supabase Storage.

NOTE: All DATABASE writes are handled by TypeScript (Next.js), NOT here.
Python only:
1. Downloads video
2. Runs analysis
3. Uploads annotated video to Supabase Storage
4. Returns results to TypeScript (which stores in DB)

Expected input:
{
    "video_url": "https://supabase.../storage/v1/object/public/videos/video.mp4",
    "stroke_type": "serve",                         # serve, groundstroke, dink, overhead, footwork, overall
    "crop_region": "0.2,0.3,0.8,0.9",              # Optional: x1,y1,x2,y2 normalized coords
    "target_point": "0.5,0.6",                      # Optional: x,y normalized coords
    "step": 3,                                      # Optional: process every Nth frame (default: 3)
    "job_id": "uuid-optional"                       # Optional: for reference only (no DB writes here)
}

Returns:
{
    "status": "success",
    "video_url": "https://supabase.../analysis-results/job123/annotated.mp4",
    "frames": [
        {"filename": "frame_0001.png", "timestampSec": 0.1, "metrics": {...}}
    ],
    "strokes": [...],
    "summary": {...},
    "injury_risk_summary": {...}
}
"""

import runpod
import os
import sys
import json
import tempfile
import shutil
import subprocess
import time
import uuid
from pathlib import Path

import requests

# Add current directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from track import NumpyEncoder
from supabase_client import get_uploader



def download_video(url: str, dest_path: str) -> bool:
    """Download video from URL to local path."""
    try:
        t0 = time.time()
        print(f"[STEP 1/5] Downloading video from: {url}")
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


def extract_frames(video_path: str, frames_dir: str, fps: int = 30) -> bool:
    """Extract frames from video using ffmpeg."""
    try:
        t0 = time.time()
        os.makedirs(frames_dir, exist_ok=True)
        
        cmd = [
            'ffmpeg', '-i', video_path,
            '-vf', f'fps={fps}',
            os.path.join(frames_dir, 'frame_%04d.png'),
            '-y', '-loglevel', 'error'
        ]
        
        print(f"[STEP 2/5] Extracting frames at {fps} FPS...")
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
        
        if result.returncode != 0:
            print(f"✗ FFmpeg error: {result.stderr}")
            return False
        
        frame_count = len(list(Path(frames_dir).glob('*.png')))
        print(f"✓ Extracted {frame_count} frames in {time.time()-t0:.2f}s")
        return frame_count > 0
    except Exception as e:
        print(f"✗ Frame extraction failed: {e}")
        return False


def encode_video_from_frames(frames_dir: str, output_path: str, fps: int = 10) -> bool:
    """Encode annotated frames back to video."""
    try:
        t0 = time.time()
        print(f"[STEP 4/5] Encoding output video...")
        cmd = [
            'ffmpeg',
            '-framerate', str(fps),
            '-i', os.path.join(frames_dir, 'frame_%04d.png'),
            '-start_number', '1',
            '-c:v', 'libx264',
            '-pix_fmt', 'yuv420p',
            '-crf', '23',
            '-preset', 'fast',
            output_path,
            '-y', '-loglevel', 'error'
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        
        if result.returncode != 0:
            print(f"✗ FFmpeg encode error: {result.stderr}")
            return False
        
        if os.path.exists(output_path):
            size_mb = os.path.getsize(output_path) / (1024 * 1024)
            print(f"✓ Video encoded: {output_path} ({size_mb:.1f} MB) in {time.time()-t0:.2f}s")
            return True
        return False
    except Exception as e:
        print(f"✗ Video encoding failed: {e}")
        return False


def run_tracking(
    input_dir: str,
    output_dir: str,
    results_json: str,
    stroke_type: str = "serve",
    crop_region: str = None,
    target_point: str = None,
    step: int = 3,
    video_path: str = None,
    analysis_windows: str = None,
    process_only_windows: bool = False,
    no_video_output: bool = False,
    no_skeleton_video: bool = False,
    coarse_mode: bool = False,
) -> dict:
    """Run the tracking pipeline."""
    try:
        t0 = time.time()
        cmd = [
            sys.executable, 'track.py',
            '--input_dir', input_dir,
            '--output_dir', output_dir,
            '--results_json', results_json,
            '--stroke_type', stroke_type,
            '--step', str(step)
        ]
        
        if crop_region:
            cmd.extend(['--crop_region', crop_region])
        if target_point:
            cmd.extend(['--target_point', target_point])
        if video_path:
            cmd.extend(['--video_path', video_path])
        if analysis_windows:
            cmd.extend(['--analysis_windows', analysis_windows])
        if process_only_windows:
            cmd.append('--process_only_windows')
        if no_video_output:
            cmd.append('--no_video_output')
        if no_skeleton_video:
            cmd.append('--no_skeleton_video')
        if coarse_mode:
            cmd.append('--coarse_mode')
        
        print(f"[STEP 3/5] Running track.py (Analysis)...")
        print(f"Command: {' '.join(cmd)}")
        
        # INCREASED TIMEOUT: 45 minutes for deep learning
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            cwd=os.path.dirname(os.path.abspath(__file__)),
            timeout=2700 
        )
        
        # STREAM LOGS: Print to RunPod logs immediately
        print("--- track.py STDOUT ---")
        print(result.stdout)
        print("--- track.py STDERR ---")
        print(result.stderr)
        print("-----------------------")
        
        if result.returncode != 0:
            print(f"✗ ERROR: track.py failed with code {result.returncode}")
            return {
                "error": f"Tracking failed with code {result.returncode}",
                "stderr": result.stderr[-2000:] if result.stderr else "No stderr output"
            }
        
        if os.path.exists(results_json):
            print(f"✓ Analysis complete in {time.time()-t0:.2f}s")
            with open(results_json, 'r') as f:
                return json.load(f)
        else:
            return {"error": "Results JSON not created"}
    
    except subprocess.TimeoutExpired:
        return {"error": "Processing timeout (exceeded 45 minutes)"}
    except Exception as e:
        return {"error": f"Tracking exception: {str(e)}"}


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
    print(f"\nPlaceholder: Starting Job {job_id}")
    print(f"Video: {video_url}")
    
    # Get parameters
    stroke_type = job_input.get("stroke_type", "serve")
    crop_region = job_input.get("crop_region")
    target_point = job_input.get("target_point")
    step = int(job_input.get("step", 1))
    
    uploader = get_uploader()
    work_dir = tempfile.mkdtemp(prefix="runpod_analysis_")
    
    try:
        # 1. Download video
        video_path = os.path.join(work_dir, "input.mp4")
        if not download_video(video_url, video_path):
            return {"error": "Failed to download video from URL"}
        
        # 2. Extract frames
        frames_dir = os.path.join(work_dir, "frames")
        if not extract_frames(video_path, frames_dir):
            return {"error": "Failed to extract frames from video"}
        
        # 3. Run tracking analysis (two-pass for long videos)
        output_root = os.path.join(work_dir, "output")
        os.makedirs(output_root, exist_ok=True)
        results_json_path = os.path.join(work_dir, "results.json")

        # Video stats
        frame_count = len(list(Path(frames_dir).glob("*.png")))
        fps = 30.0
        duration_sec = frame_count / fps if frame_count > 0 else 0.0

        use_two_pass = duration_sec >= 20.0
        results = None

        if use_two_pass:
            # PASS 1: Coarse scan (cheap) + annotated video
            pass1_dir = os.path.join(output_root, "pass1")
            os.makedirs(pass1_dir, exist_ok=True)
            pass1_json = os.path.join(work_dir, "results_pass1.json")
            results_pass1 = run_tracking(
                input_dir=frames_dir,
                output_dir=pass1_dir,
                results_json=pass1_json,
                stroke_type=stroke_type,
                crop_region=crop_region,
                target_point=target_point,
                step=5,
                video_path=video_path,
                no_skeleton_video=True,
                coarse_mode=True,
            )
            if "error" in results_pass1:
                return results_pass1

            # Build candidate windows from pass1 strokes
            strokes1 = results_pass1.get("strokes", []) or []
            margin = 1.5  # seconds around coarse detection
            windows = []
            for s in strokes1:
                s0 = float(s.get("startSec", 0.0) or 0.0)
                s1 = float(s.get("endSec", s0) or s0)
                w0 = max(0.0, s0 - margin)
                w1 = min(duration_sec, s1 + margin)
                if w1 > w0:
                    windows.append((w0, w1))
            windows.sort()
            merged = []
            for w0, w1 in windows:
                if not merged or w0 > merged[-1][1] + 0.25:
                    merged.append([w0, w1])
                else:
                    merged[-1][1] = max(merged[-1][1], w1)
            analysis_windows = ",".join([f"{a:.2f}:{b:.2f}" for a, b in merged])

            # PASS 2: Refined analysis only inside windows (accurate)
            pass2_dir = os.path.join(output_root, "pass2")
            os.makedirs(pass2_dir, exist_ok=True)
            pass2_json = os.path.join(work_dir, "results_pass2.json")
            results_pass2 = run_tracking(
                input_dir=frames_dir,
                output_dir=pass2_dir,
                results_json=pass2_json,
                stroke_type=stroke_type,
                crop_region=crop_region,
                target_point=target_point,
                step=1,
                video_path=video_path,
                analysis_windows=analysis_windows if analysis_windows else None,
                process_only_windows=bool(analysis_windows),
                no_video_output=True,
                no_skeleton_video=True,
                coarse_mode=False,
            )
            if "error" in results_pass2:
                return results_pass2

            # Merge outputs:
            # - Use refined strokes/frames from pass2
            # - Use full-duration summary from pass1
            merged_out = dict(results_pass2)
            merged_out["summary"] = results_pass1.get("summary", merged_out.get("summary", {})) or {}
            merged_out["summary"]["tracked_duration_sec"] = round(duration_sec, 2)
            merged_out["summary"]["fps"] = int(fps)
            if not merged_out.get("strokes"):
                merged_out["strokes"] = strokes1

            # Write final merged results.json for upload
            with open(results_json_path, "w") as f:
                json.dump(merged_out, f, indent=2, cls=NumpyEncoder)

            results = merged_out

            # Encode annotated video from pass1 outputs
            annotated_video_path = os.path.join(pass1_dir, "annotated.mp4")
            video_encoded = encode_video_from_frames(pass1_dir, annotated_video_path, fps=max(1, int(fps // 5)))
        else:
            # Single pass (short videos)
            output_dir = os.path.join(output_root, "single")
            os.makedirs(output_dir, exist_ok=True)
            results = run_tracking(
                input_dir=frames_dir,
                output_dir=output_dir,
                results_json=results_json_path,
                stroke_type=stroke_type,
                crop_region=crop_region,
                target_point=target_point,
                step=step,
                video_path=video_path,
                no_skeleton_video=True,
            )
            if "error" in results:
                return results

            annotated_video_path = os.path.join(output_dir, "annotated.mp4")
            output_fps = max(1, 30 // step)
            video_encoded = encode_video_from_frames(output_dir, annotated_video_path, fps=output_fps)
        
        # 5. Upload to Supabase
        print(f"[STEP 5/5] Uploading results...")
        result_video_url = None
        frames_data = []
        uploaded_frame_urls = {}
        
        if video_encoded and os.path.exists(annotated_video_path):
            print(f"Uploading annotated video...")
            result_video_url = uploader.upload_file(
                bucket="analysis-results",
                file_path=annotated_video_path,
                destination_path=f"{job_id}/annotated.mp4",
                content_type="video/mp4"
            )

        # Upload skeleton video (DISABLED to reduce RunPod completion time + storage)
        skeleton_video_url = None

        # Upload full results JSON (python metrics/frames/summary) for debugging + UI fallback
        results_json_url = None
        if os.path.exists(results_json_path):
            print("Uploading results.json...")
            results_json_url = uploader.upload_file(
                bucket="analysis-results",
                file_path=results_json_path,
                destination_path=f"{job_id}/results.json",
                content_type="application/json"
            )

        # Key-frame upload (DISABLED to reduce RunPod completion time + storage)
        # uploaded_frame_urls stays empty; frameFilename will be None.
        
        # Build frames response
        # IMPORTANT:
        # - frameIdx: analyzed-frame index (0..N-1)
        # - frame_idx: original video frame index (0..videoFrames-1)
        # Key-frame uploads are keyed by ORIGINAL frame index, so URLs should map via frame_idx.
        raw_frames = results.get("frames", [])
        fps = (results.get("summary") or {}).get("fps", 30) or 30
        for i, frame_data in enumerate(raw_frames):
            analyzed_idx = frame_data.get("frameIdx", i)
            original_idx = frame_data.get("frame_idx", analyzed_idx)
            frame_url = uploaded_frame_urls.get(original_idx)
            frames_data.append({
                "frameIndex": analyzed_idx,
                "frame_idx": original_idx,
                "timestampSec": frame_data.get("timestampSec", float(original_idx) / float(fps)),
                "bbox": frame_data.get("bbox", [0, 0, 0, 0]),
                "confidence": frame_data.get("confidence", 0),
                "metrics": frame_data.get("metrics", {}),
                # Preserve MediaPipe landmarks for TypeScript analyzeFrames fallback (webhook side)
                "landmarks": frame_data.get("landmarks", None),
                # URL of uploaded keyframe (if this frame_idx was uploaded); otherwise None
                "frameFilename": frame_url
            })
        
        processing_time = time.time() - start_time
        print(f"=== JOB COMPLETE: {job_id} in {processing_time:.1f}s ===")
        
        return {
            "status": "success",
            "job_id": job_id,
            "video_url": result_video_url,
            "skeleton_video_url": skeleton_video_url,
            "results_json_url": results_json_url,
            "frames": frames_data,
            "strokes": results.get("strokes", []),
            "summary": results.get("summary", {}),
            "injury_risk_summary": results.get("injury_risk_summary", {}),
            "total_frames_processed": len(frames_data),
            "processing_time_sec": round(processing_time, 2)
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
    print("StrikeSense Video Analysis Worker")
    print("=" * 60)
    print(f"Supabase URL: {os.environ.get('SUPABASE_URL', 'NOT SET')}")
    print(f"Supabase Key: {'SET' if os.environ.get('SUPABASE_SERVICE_KEY') else 'NOT SET'}")
    print("=" * 60)
    
    # Startup diagnostics
    print("\n--- Startup Diagnostics ---")
    print(f"Working directory: {os.getcwd()}")
    print(f"Python executable: {sys.executable}")
    print(f"PYTHONPATH: {os.environ.get('PYTHONPATH', 'NOT SET')}")
    
    # Check models folder
    models_dir = os.path.join(os.path.dirname(__file__), 'models')
    if os.path.exists(models_dir):
        print(f"Models directory: {models_dir}")
        print(f"Models found: {os.listdir(models_dir)}")
    else:
        print(f"WARNING: Models directory not found at {models_dir}")
    
    # Test track.py imports
    print("\n--- Testing track.py imports ---")
    try:
        import track
        print("✓ track.py imported successfully")
    except Exception as e:
        print(f"✗ track.py import failed: {e}")
    
    try:
        from ultralytics import YOLO
        print("✓ ultralytics (YOLO) imported successfully")
    except Exception as e:
        print(f"✗ ultralytics import failed: {e}")
    
    try:
        import mediapipe as mp
        print("✓ mediapipe imported successfully")
    except Exception as e:
        print(f"✗ mediapipe import failed: {e}")

    try:
        import torch
        print(f"✓ torch imported: {torch.__version__}")
        print(f"  CUDA Available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"  GPU: {torch.cuda.get_device_name(0)}")
    except ImportError:
        print("✗ torch import failed")
    
    print("=" * 60)
    print("Starting RunPod Serverless Worker...")
    print("=" * 60 + "\n")
    
    runpod.serverless.start({"handler": handler})
