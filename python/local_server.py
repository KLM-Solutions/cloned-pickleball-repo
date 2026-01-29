"""
Local Development Server for Pickleball Video Analysis

This Flask server mimics RunPod's serverless API for local development.
Run this instead of deploying to RunPod when testing locally.

Usage:
    cd python
    & .\venv\Scripts\Activate.ps1
    python local_server.py

The server will run on http://localhost:8000
"""

from flask import Flask, request, jsonify
import threading
import time
import uuid
import os
import sys
import json
import requests

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import the handler from handler.py
from handler import handler

app = Flask(__name__)

# Store job statuses (in-memory for development)
jobs = {}

def process_job_async(job_id, job_input, webhook_url=None):
    """Process job in background thread and optionally call webhook."""
    try:
        jobs[job_id]["status"] = "IN_PROGRESS"
        
        # Create job structure like RunPod
        job = {
            "id": job_id,
            "input": job_input
        }
        
        # Run the actual handler
        result = handler(job)
        
        # Store result
        jobs[job_id]["status"] = "COMPLETED"
        jobs[job_id]["output"] = result
        
        # Call webhook if provided
        if webhook_url:
            try:
                webhook_payload = {
                    "id": job_id,
                    "status": "COMPLETED",
                    "output": result,
                    "input": job_input
                }
                print(f"Calling webhook: {webhook_url}")
                response = requests.post(
                    webhook_url, 
                    json=webhook_payload,
                    timeout=30
                )
                print(f"Webhook response: {response.status_code}")
            except Exception as e:
                print(f"Webhook call failed: {e}")
                
    except Exception as e:
        jobs[job_id]["status"] = "FAILED"
        jobs[job_id]["error"] = str(e)
        print(f"Job {job_id} failed: {e}")

@app.route("/run", methods=["POST"])
def run_async():
    """
    Start an async job (like RunPod /run endpoint).
    Returns job ID immediately, processes in background.
    """
    data = request.json
    job_id = str(uuid.uuid4())
    job_input = data.get("input", {})
    webhook_url = data.get("webhook")
    
    # Initialize job status
    jobs[job_id] = {
        "status": "IN_QUEUE",
        "output": None,
        "error": None
    }
    
    # Start processing in background
    thread = threading.Thread(
        target=process_job_async,
        args=(job_id, job_input, webhook_url)
    )
    thread.start()
    
    return jsonify({
        "id": job_id,
        "status": "IN_QUEUE"
    })

@app.route("/runsync", methods=["POST"])
def run_sync():
    """
    Run job synchronously (like RunPod /runsync endpoint).
    Waits for completion and returns result.
    """
    data = request.json
    job_id = str(uuid.uuid4())
    job_input = data.get("input", {})
    
    # Create job structure
    job = {
        "id": job_id,
        "input": job_input
    }
    
    # Run handler directly
    try:
        result = handler(job)
        return jsonify({
            "id": job_id,
            "status": "COMPLETED",
            "output": result
        })
    except Exception as e:
        return jsonify({
            "id": job_id,
            "status": "FAILED",
            "error": str(e)
        }), 500

@app.route("/status/<job_id>", methods=["GET"])
def get_status(job_id):
    """Check job status (like RunPod /status endpoint)."""
    if job_id not in jobs:
        return jsonify({"error": "Job not found"}), 404
    
    job = jobs[job_id]
    response = {
        "id": job_id,
        "status": job["status"]
    }
    
    if job["status"] == "COMPLETED":
        response["output"] = job["output"]
    elif job["status"] == "FAILED":
        response["error"] = job["error"]
    
    return jsonify(response)

@app.route("/cancel/<job_id>", methods=["POST"])
def cancel_job(job_id):
    """Cancel a job (placeholder - not fully implemented)."""
    if job_id in jobs:
        jobs[job_id]["status"] = "CANCELLED"
    return jsonify({"status": "cancelled"})

@app.route("/health", methods=["GET"])
def health():
    """Health check endpoint."""
    return jsonify({
        "status": "healthy",
        "jobs_in_memory": len(jobs)
    })

if __name__ == "__main__":
    print("=" * 60)
    print("StrikeSense Local Development Server")
    print("=" * 60)
    print(f"Server running at: http://localhost:8000")
    print(f"API Endpoints:")
    print(f"  POST /run      - Start async job")
    print(f"  POST /runsync  - Run sync job")
    print(f"  GET  /status/  - Check job status")
    print(f"  GET  /health   - Health check")
    print("=" * 60)
    
    # Run Flask server
    app.run(host="0.0.0.0", port=8000, debug=False, threaded=True)
