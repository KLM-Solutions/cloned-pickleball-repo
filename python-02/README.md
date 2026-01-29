# MediaPipe Video Annotation Server

Standalone Python server for video pose analysis using MediaPipe.

## Structure

```
python-02/
├── handler.py          # RunPod serverless handler
├── local_server.py     # Flask server for local dev
├── track.py            # MediaPipe processing
├── supabase_client.py  # Supabase upload
├── verify_env.py       # Environment verification
├── requirements.txt    # Dependencies
├── Dockerfile.runpod   # RunPod deployment
├── .env.example        # Environment template
└── README.md           # This file
```

## Local Development

```bash
# Setup
python -m venv venv
.\venv\Scripts\Activate.ps1  # Windows
pip install -r requirements.txt

# Configure
cp .env.example .env
# Edit .env with Supabase credentials

# Verify
python verify_env.py

# Run
python local_server.py
```

## RunPod Deployment

```bash
# Build
docker build -f Dockerfile.runpod -t yourusername/mediapipe-worker:latest .

# Push
docker push yourusername/mediapipe-worker:latest
```

In RunPod:
- Docker Image: `yourusername/mediapipe-worker:latest`
- Environment: `SUPABASE_URL`, `SUPABASE_SERVICE_KEY`

## API

### POST /runsync (local) or RunPod Job

**Input:**
```json
{
  "input": {
    "video_url": "https://...",
    "output_bucket": "annotated-videos",
    "job_id": "optional-uuid"
  }
}
```

**Output:**
```json
{
  "status": "success",
  "job_id": "...",
  "annotated_video_url": "https://...",
  "landmarks": [...],
  "metadata": {"fps": 30, "total_frames": 150, "duration_sec": 5.0}
}
```
