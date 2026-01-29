"""
Generic MediaPipe Video Annotation Server

A Flask server that processes videos with MediaPipe pose detection,
generates annotated videos with skeleton overlays, and uploads to Supabase.

Usage:
    python server.py

Environment variables (.env):
    SUPABASE_URL - Your Supabase project URL
    SUPABASE_SERVICE_KEY - Service role key
    PORT - Server port (default: 8000)
"""

import os
import sys
import json
import uuid
import time
import tempfile
import shutil
from pathlib import Path

from flask import Flask, request, jsonify
from flask_cors import CORS
from dotenv import load_dotenv
import requests

# Load environment variables
load_dotenv()

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from annotator import VideoAnnotator
from supabase_client import get_uploader

app = Flask(__name__)
CORS(app)


def download_video(url: str, dest_path: str) -> bool:
    """Download video from URL to local path."""
    try:
        print(f"Downloading video from: {url}")
        response = requests.get(url, stream=True, timeout=300)
        response.raise_for_status()
        
        with open(dest_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        
        size_mb = os.path.getsize(dest_path) / (1024 * 1024)
        print(f"Downloaded: {size_mb:.1f} MB")
        return True
    except Exception as e:
        print(f"Download failed: {e}")
        return False


@app.route("/analyze", methods=["POST"])
def analyze():
    """
    Analyze video with MediaPipe pose detection.
    
    Request JSON:
        video_url: Supabase storage URL to input video
        output_bucket: Bucket name for annotated video output
    
    Returns JSON:
        status: "success" or "error"
        annotated_video_url: URL of annotated video
        landmarks: Array of per-frame pose landmarks
        metadata: Video metadata (fps, duration, etc.)
    """
    start_time = time.time()
    
    # Parse request
    data = request.get_json()
    if not data:
        return jsonify({"status": "error", "message": "No JSON body"}), 400
    
    video_url = data.get("video_url")
    output_bucket = data.get("output_bucket", "annotated-videos")
    job_id = data.get("job_id") or str(uuid.uuid4())  # Use provided ID or generate new
    
    if not video_url:
        return jsonify({"status": "error", "message": "Missing video_url"}), 400
    work_dir = tempfile.mkdtemp(prefix="mediapipe_")
    
    try:
        # 1. Download video
        input_path = os.path.join(work_dir, "input.mp4")
        if not download_video(video_url, input_path):
            return jsonify({"status": "error", "message": "Failed to download video"}), 400
        
        # 2. Process with MediaPipe
        print("Processing with MediaPipe...")
        output_path = os.path.join(work_dir, "annotated.mp4")
        
        annotator = VideoAnnotator()
        landmarks, metadata = annotator.process_video(input_path, output_path)
        annotator.close()
        
        print(f"Processed {metadata['total_frames']} frames")
        
        # 3. Upload annotated video
        uploader = get_uploader()
        annotated_url = None
        
        if os.path.exists(output_path):
            annotated_url = uploader.upload_file(
                bucket=output_bucket,
                file_path=output_path,
                destination_path=f"{job_id}.mp4",
                content_type="video/mp4"
            )
        
        processing_time = round(time.time() - start_time, 2)
        
        return jsonify({
            "status": "success",
            "job_id": job_id,
            "annotated_video_url": annotated_url,
            "landmarks": landmarks,
            "metadata": {
                **metadata,
                "processing_time_sec": processing_time
            }
        })
    
    except Exception as e:
        print(f"Error: {e}")
        return jsonify({"status": "error", "message": str(e)}), 500
    
    finally:
        # Cleanup
        try:
            shutil.rmtree(work_dir)
        except:
            pass


@app.route("/health", methods=["GET"])
def health():
    """Health check endpoint."""
    return jsonify({
        "status": "healthy",
        "supabase_configured": bool(os.environ.get("SUPABASE_URL"))
    })


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    
    print("=" * 50)
    print("MediaPipe Video Annotation Server")
    print("=" * 50)
    print(f"Port: {port}")
    print(f"Supabase URL: {os.environ.get('SUPABASE_URL', 'NOT SET')}")
    print("=" * 50)
    
    app.run(host="0.0.0.0", port=port, debug=False)
