# StrikeSense: Python Computer Vision Architecture

This document provides a deep dive into the **AI/Computer Vision Backend** of StrikeSense. It details exactly how raw video is transformed into biomechanical insights.

## directory Structure
```
python/
├── track.py                # MAIN ENTRY POINT (The Orchestrator)
├── biomechanics/           # PHYSICS ENGINE
│   ├── angles.py           # Geometry logic (Calculation of 2D/3D angles)
│   └── analyzer.py         # Risk detection based on angles
├── classification/         # INTELLIGENCE ENGINE
│   ├── classifier.py       # Heuristic Logic to name strokes (Serve vs Dink)
│   └── rules.py            # Definition of stroke signatures
└── requirements.txt        # Dependencies (YOLO, MediaPipe, OpenCV)
```

---

## The Computer Vision Pipeline (End-to-End)

We process every video in **5 Distinct Layers**.

### Layer 1: Ingestion & Optimization
**Goal**: Prepare the data for fast processing.
1.  **Read Frame**: `cv2.VideoCapture` loads the video frame.
2.  **Crop (The Speedup)**: If the user defined a crop box on the frontend, we **discard 70% of the pixels** immediately. We only feed the "Active Play Area" to the expensive AI models. This is why the app is responsive.

### Layer 2: Object Detection (YOLOv8)
**Goal**: Find the human in the chaos.
-   **Model**: YOLOv8 Nano (`yolov8n.pt`) - Small, fast, accurate.
-   **Why**: MediaPipe (the Skeleton AI) is sensitive. If we run it on an empty court or a spectator, it fails.
-   **Method**:
    1.  Run YOLO on the cropped frame.
    2.  Filter for `class_id == 0` (Person).
    3.  **BoxMOT Tracking**: We assign a specific ID (e.g., "ID 1") to the player. If they walk behind a net post and come out, BoxMOT remembers "That's still Player 1".

### Layer 3: Pose Estimation (MediaPipe)
**Goal**: Extract the structure of the body.
-   **Model**: Google MediaPipe Pose (BlazePose).
-   **Output**: 33 Keypoints (X, Y, Z, Visibility).
-   **Crucial Keypoints for Pickleball**:
    -   11/12 (Shoulders)
    -   13/14 (Elbows)
    -   15/16 (Wrists)
    -   23/24 (Hips)
    -   25/26 (Knees)

### Layer 4: Biomechanical Math (The "Sense" in StrikeSense)
**Goal**: Convert abstract points into physics data.
-   **File**: `biomechanics/angles.py`
-   **Logic**:
    -   **Shoulder Abduction**: Calculating the 2D angle between the *Torso Axis* (Hip-to-Shoulder) and the *Arm Axis* (Shoulder-to-Elbow).
        -   *Math*: Law of Cosines or Vector Dot Product.
    -   **Knee Flexion**: Angle between Hip-Knee-Ankle.
    -   **Risk Assessment**: The `InjuryRiskAnalyzer` compares these calculated angles against medical thresholds.
        -   *Rule*: If Shoulder Abduction > 120° repeatedly -> Flag "Impingement Risk".

### Layer 5: Temporal Classification (Stroke Recognition)
**Goal**: Understand *what* movement just happened.
-   **Problem**: A single frame looks the same for a High-Five and a Serve. We need **Time**.
-   **Method**: **Heuristic State Machine**.
    1.  **Buffer**: We keep the last 30 frames of Wrist & Elbow positions in memory.
    2.  **Signature Matching**:
        -   **SERVE**: `Wrist Height` goes from Low -> High + `Velocity` is Fast + `Frame Position` is Start of Point.
        -   **DINK**: `Wrist Height` stays Low + `Velocity` is Slow + `Arm Extension` is Low.
        -   **GROUNDSTROKE**: `Wrist` moves horizontally across the body (Right -> Left for right-hander).

---

## Why This Approach? (Trade-offs)

| Approach | Pros | Cons | our Choice |
| :--- | :--- | :--- | :--- |
| **End-to-End Deep Learning (VideoMAE)** | Extremely accurate classification. | Requires massive GPU (A100). Slow. Data hungry. | ❌ No |
| **Heuristic Hybrid (YOLO + Pose + Math)** | **Fast (CPU capable)**. Interpretability (we know *why* it decided "Serve"). | Hard to code complex rules. 2D perspective issues. | ✅ **YES** |

## Data Flow Summary
1.  **Video File** (`.mp4`)  --> **Input**
2.  **Pixels** (`numpy array`) --> **YOLO** --> **Bounding Box**
3.  **Bounding Box** --> **MediaPipe** --> **Keypoints (X,Y,Z)**
4.  **Keypoints** --> **Math Engine** --> **Angles (Degrees)**
5.  **Angles + Time** --> **Classifier** --> **Stroke Name ("Forehand")**
6.  **Results** --> **JSON File** --> **Frontend Dashboard**
