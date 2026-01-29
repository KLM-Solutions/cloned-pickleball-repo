# MediaPipe Video Annotation Server

A standalone Python server that processes videos with MediaPipe pose detection, generates annotated videos with skeleton overlays, and stores results in Supabase.

## Setup

```bash
# Create virtual environment
python -m venv venv

# Activate (Windows)
.\venv\Scripts\Activate.ps1

# Activate (Linux/Mac)
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

## Configuration

Copy `.env.example` to `.env` and fill in your Supabase credentials:

```env
SUPABASE_URL=https://your-project.supabase.co
SUPABASE_SERVICE_KEY=your-service-role-key
PORT=8000
```

## Run Server

```bash
python server.py
```

Server runs at `http://localhost:8000`

## API

### `POST /analyze`

Process video with MediaPipe pose detection.

**Request:**
```json
{
  "video_url": "https://<project>.supabase.co/storage/v1/object/public/videos/input.mp4",
  "output_bucket": "annotated-videos"
}
```

**Response:**
```json
{
  "status": "success",
  "job_id": "uuid",
  "annotated_video_url": "https://.../annotated-videos/<job_id>.mp4",
  "landmarks": [
    {
      "frame_index": 0,
      "timestamp_sec": 0.033,
      "pose": [
        {"x": 0.52, "y": 0.31, "z": -0.12, "visibility": 0.99},
        ...
      ]
    }
  ],
  "metadata": {
    "fps": 30,
    "total_frames": 150,
    "duration_sec": 5.0
  }
}
```

### `GET /health`

Health check endpoint.

## Landmark Reference

MediaPipe returns 33 pose landmarks per frame:

| Index | Landmark |
|-------|----------|
| 0 | Nose |
| 11-12 | Shoulders |
| 13-14 | Elbows |
| 15-16 | Wrists |
| 23-24 | Hips |
| 25-26 | Knees |
| 27-28 | Ankles |

Full list: [MediaPipe Pose Landmarks](https://developers.google.com/mediapipe/solutions/vision/pose_landmarker)
