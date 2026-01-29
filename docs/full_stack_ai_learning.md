# StrikeSense: Full Stack + AI Architecture Learning Guide

This guide breaks down the 7-step workflow of the application, explaining how the **Frontend (Next.js/React)** interacts with the **AI Backend (Python/Computer Vision)**.

## 1. Landing Page (`/`)
**The Entry Point: Setting Expectations**
-   **Frontend UX**: A high-performance React marketing page that instantly establishes credibility.
-   **AI Capabilities Advertised**: We explicitly list the **6 Key Strokes** our AI model is trained to recognize. This isn't just marketing; it maps directly to the `StrokeClassifier` class in Python.
    1.  **Serve** 🎾: Recognized by high extension and initial velocity.
    2.  **Forehand Groundstroke** 💪: The most common shot; detected by open stance and swing path.
    3.  **Backhand Groundstroke** 💪: Identified by cross-body arm movement.
    4.  **Volley** 🏓: Short, punchy motions detected near the net (if depth logic is active).
    5.  **Dink** 🤏: Soft, controlled arcs with minimal racket speed.
    6.  **Overhead (Smash)** ⚡: High vertical extension and rapid downward acceleration.
-   **Full Stack Concept**: **Routing**. When "Start Analysis" is clicked, Next.js performs a *Client-Side Transition* to `/strikesense/upload`, pre-fetching the code for that page instantly.

## 2. Upload Page (`/strikesense/upload`)
**Handling Large Files Efficiently**
-   **Frontend Tech**: HTML5 `File` API + Drag-and-Drop events.
-   **The Problem**: Sending a 500MB 4K video to a server immediately is slow and expensive.
-   **The Solution (MVP)**: We use **Client-Side Blob URLs**.
    -   `URL.createObjectURL(file)` creates a temporary link to the file *staying on your computer's RAM*.
    -   We pass this "virtual link" to the next steps. No server upload happens yet!
-   **Browser Storage**: For persistence across refreshes, we store the file handle in `IndexedDB` (a mini database inside your browser).

## 3. Camera Guide (`/strikesense/guide`)
**Data Normalization (The Secret Sauce)**
-   **AI Principle**: **Garbage In, Garbage Out**.
    -   Our Python model (`pose.process()`) is finding X,Y coordinates on a 2D image.
    -   If a user records from the *front*, a 90° arm angle looks like 45°. The math breaks.
-   **UX Enforcement**: We force the user to acknowledge the **"90° Side Profile"** requirement.
    -   This ensures the `right_shoulder`, `right_elbow`, and `right_wrist` keypoints are clearly visible and planar to the camera, maximizing accuracy for all 6 strokes.

## 4. Crop Page (`/strikesense/crop`)
**Region of Interest (ROI) Optimization**
-   **Frontend Feature**: Interactive canvas overlay where users draw a bounding box.
-   **Technical Backend Impact**:
    -   **Speed**: Processing a 1080p frame (1920x1080 pixels) takes ~100ms. Processing a cropped 500x500 box takes ~20ms. That's a **5x speedup**.
    -   **Accuracy**: By cropping out the background (spectators, trees), the YOLO person detector (`model.predict()`) won't accidentally lock onto a person walking in the background. It focuses purely on the player.

## 5. Processing Page (`/strikesense/processing`)
**The Bridge between Web & AI**
-   **The "Black Box" Workflow**:
    1.  **Input**: The video file + Crop Coordinates.
    2.  **Python Script (`track.py`)**:
        -   **OpenCV**: Reads the video frame by frame.
        -   **YOLOv8 & BoxMOT**: Tracks the specific player ID across frames (ignoring others).
        -   **MediaPipe**: Extracts 33 skeleton keypoints (X,Y,Z coordinates) for every single frame.
    3.  **Classification Logic**:
        -   The script analyzes the *trajectory* of the wrist.
        -   *Upward Arc + High Speed* = **Serve/Overhead**.
        -   *Horizontal S-Curve* = **Groundstroke**.
        -   *Small Arc + Low Speed* = **Dink**.
    4.  **Output**: A simplistic JSON file (`analysis.json`) containing timestamps, stroke types, and risk scores.

## 6. Player Page (`/strikesense/player`)
**Data Visualization & Synchronization**
-   **The Hardest Part**: Syncing 60 lines of data per second with a playing video.
-   **Technique**: `requestAnimationFrame` loop.
    -   **Step 1**: Current Video Time is `3.54s`.
    -   **Step 2**: React looks up the JSON data found at index `3.54`.
    -   **Step 3**: HTML5 Canvas draws the skeleton lines *over* the video element instantly.
-   **Interactivity**: When you click a "Forehand" in the list, React sets `video.currentTime = stroke.timestamp`, instantly jumping you to that moment.

## 7. Analysis Page (`/strikesense/analysis`)
**Aggregated Intelligence**
-   **From Raw Data to Wisdom**:
    -   **Raw**: "Elbow angle is 170 degrees at frame 405."
    -   **Insight**: "Your elbow hyperextends on 80% of Backhands -> High Injury Risk."
-   **Components**:
    -   **Risk Gauges**: Visualizes average joint stress. If the average shoulder angle > 110° (Safe limit), the needle moves to Red.
    -   **Stroke Breakdown**: Counts `len(strokes)` classified as "Dink" vs "Volley" to give strategic feedback (e.g., "You play too aggressively at the net").

---

## Summary of the Stack
-   **Frontend**: Next.js (The Face) - Handles user interaction, file management, and visualization.
-   **Backend**: Python (The Brain) - Uses Computer Vision to extract structure from pixels.
-   **Protocol**: JSON (The Language) - How the Face and Brain talk to each other.
