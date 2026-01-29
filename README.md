# Strike Sense: AI Pickleball Analysis

Strike Sense is an advanced AI-powered application for analyzing Pickleball gameplay. It uses computer vision (YOLOv8, MediaPipe) to track players, analyze biomechanics, and classify strokes (Serves, Volleys, Groundstrokes) to provide injury risk assessments and performance metrics.

## 🚀 Features

*   **Intelligent Player Tracking**: Robust tracking of players using YOLOv8 + DeepOCSORT, with re-identification logic to handle frame exits and occlusions.
*   **Biomechanics Analysis**: real-time analysis of joint angles (Elbow, Shoulder, Knee, Hip) using MediaPipe Pose Estimation.
*   **Stroke Classification**: Automatically detects and classifies common pickleball strokes.
*   **Injury Risk Assessment**: Calculates injury risks based on biomechanical thresholds.
*   **Dynamic Cropping**: Automatically zooms and focuses on the active player for detailed analysis.
*   **Video Annotation**: Generates high-quality output videos with skeletal overlays and metric dashboards.
*   **Modern UI**: Sleek, responsive Next.js frontend with "Energetic Teal" and "Deep Navy" theme.

## 🛠️ Tech Stack

*   **Frontend**: Next.js 14, React, Tailwind CSS, Lucide Icons.
*   **Backend / AI Engine**: Python 3.11+
*   **Computer Vision**:
    *   Ultralytics YOLOv8 (Person Detection)
    *   MediaPipe (Pose Estimation)
    *   BoxMOT (DeepOCSORT Tracking)
    *   OpenCV (Video Processing)
*   **Data Serialization**: Custom NumPy encoders for robust JSON API responses.

## 📦 Installation

### Prerequisites
*   Node.js 18+
*   Python 3.10 or 3.11 (Python 3.12+ may have compatibility issues with some ML libraries)

### 1. Clone the Repository
```bash
git clone https://github.com/KLM-Solutions/pickle-ball.git
cd pickle-ball
```

### 2. Frontend Setup
```bash
npm install
```

### 3. Backend (Python) Setup
The application is designed to use a global Python installation or a virtual environment.

**Option A (Recommended): Create a Virtual Environment**
```bash
# Windows
python -m venv .venv
.venv\Scripts\activate

# Mac/Linux
python3 -m venv .venv
source .venv/bin/activate

# Install Dependencies
pip install -r python/requirements.txt
```

**Option B: Global Installation**
Ensure all requirements are installed in your global Python:
```bash
pip install -r python/requirements.txt
```

## ▶️ Usage

### Running Development Server
You can start the full stack (Frontend + Backend API) with one command:

```bash
npm run dev
```

*   Frontend: `http://localhost:3000`
*   The application will automatically locate your Python environment (checking `.venv`, `venv`, or global `python`).

### How to Analyze a Video
1.  Navigate to the **Upload** page.
2.  Select a pickleball video file (`.mp4`, `.mov`).
3.  Choose the **Stroke Type** you want to analyze (e.g., Serve, Volley).
4.  **Crop**: Draw a bounding box around the player you want to track.
5.  **Analyze**: The AI engine will process the video.
6.  **Results**: View the annotated video, biomechanics charts, and frame-by-frame breakdown.

## 📂 Project Structure

```
├── app/                 # Next.js Frontend Pages & API Routes
│   ├── api/analyze/     # Main Analysis Endpoint
│   ├── strikesense/     # UI Pages (Upload, Crop, Processing, Result)
├── python/              # AI/ML Backend Logic
│   ├── track.py         # Main Tracking & Analysis Script
│   ├── biomechanics/    # Biomechanics Calculation Modules
│   ├── classification/  # Stroke Classification Logic
│   └── requirements.txt # Python Dependencies
├── public/              # Static Assets
└── components/          # Reusable React Components
```

## 🤝 Contributing
1.  Fork the repo
2.  Create your feature branch (`git checkout -b feature/amazing-feature`)
3.  Commit your changes (`git commit -m 'Add some amazing feature'`)
4.  Push to the branch (`git push origin feature/amazing-feature`)
5.  Open a Pull Request
