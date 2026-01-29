# Research: Next-Gen Computer Vision Models (YOLOv11 & ViTPose)

**Objective**: Analyze the potential of upgrading the current stack (YOLOv8 + MediaPipe) to state-of-the-art models (YOLOv11, ViTPose).

---

## 1. YOLOv11 (The Speed Demon)
**What is it?**: The latest iteration from Ultralytics (released late 2024), optimizing the YOLO architecture for even faster inference and higher accuracy.

### Key Metrics vs. YOLOv8 (Our Current Model)
| Metric | YOLOv8n (Current) | YOLOv11n (Upgrade) | Improvement |
| :--- | :--- | :--- | :--- |
| **Speed (CPU)** | ~80ms / frame | ~55ms / frame | **~30% Faster** 🚀 |
| **Accuracy (mAP)** | 37.3 (COCO) | 39.5 (COCO) | **+2.2% Accuracy** |
| **Parameters** | 3.2M | 2.6M | **18% Smaller** |

### Potential Impact on StrikeSense
-   **Processing Speed**: Switching the detection step to YOLOv11 would significantly reduce the time users wait on the "Processing..." screen.
-   **Accuracy**: Better detection of small objects or players in bad lighting.
-   **The "Pose" Variant**: YOLOv11 also has a `yolo11-pose.pt` model.
    -   *Pros*: Fast multi-person pose estimation.
    -   *Cons*: Only detects **17 Keypoints** (COCO Standard). MediaPipe detects **33 Keypoints** (including hands/feet detail).
    -   *Verdict*: Moving identifying logic (Person Detection) to YOLOv11 is a **No-Brainer**. Moving Pose logic is a **Downgrade** for biomechanics detail.

---

## 2. ViTPose (The Accuracy King)
**What is it?**: "Vision Transformer for Pose Estimation". It uses Transformer architecture (like GPT but for images) instead of standard CNNs.

### Comparison
| Feature | MediaPipe (Current) | ViTPose (Research) |
| :--- | :--- | :--- |
| **Architecture** | CNN / MobileNet (Lightweight) | Vision Transformer (Heavy) |
| **Accuracy** | Good (~80-85% PCK) | **State of the Art (>90% PCK)** 🏆 |
| **Hardware** | Runs on **CPU** (Mobile/Laptop) | Requires **Heavy GPU** (Server) |
| **Keypoints** | 33 (Full Body + Hands) | Varies (Usually 17 or 25) |
| **Speed** | 30+ FPS (Real-time) | ~5-10 FPS (Slow via API) |

### Potential Impact on StrikeSense
-   **Biomechanical Precision**: ViTPose would eliminate "jitter" (shaky skeleton lines). Angles would be medically accurate.
-   **Cost/Complexity**: We cannot run this in the browser or on Vercel. It would require a dedicated GPU server ($$$).
-   **"ViPose" Note**: There is also a niche research paper called "ViPose" regarding visibility-aware pose estimation, but **ViTPose** is the industry standard you likely mean.

---

## 3. Recommendation for StrikeSense

### Immediate Win (Low Effort, High Reward)
**Upgrade Object Detection to YOLOv11.**
-   **Why**: It's a drop-in replacement for YOLOv8.
-   **Gain**: 30% faster processing speed for the user.
-   **Cost**: Free (Open Source).

### Strategic Pivot (High Effort, High Quality)
**Keep MediaPipe for MVP, but explore specific "Sport-Fine-Tuned" models.**
-   **Why**: Replacing MediaPipe with YOLO-Pose loses key skeletal details (33 vs 17 points). Replacing it with ViTPose requires expensive GPUs.
-   **Best Path**: Stick with MediaPipe for the Web/Mobile MVP because it runs *everywhere* for free.

---

## Summary Table

| Stack Component | Current Implementation | Proposed Upgrade | Expected Result |
| :--- | :--- | :--- | :--- |
| **Person Detection** | YOLOv8 Nano | **YOLOv11 Nano** | **30% Faster processing** |
| **Skeleton Tracking** | MediaPipe BlazePose | **MediaPipe (Keep)** | Best balance of free/accuracy. |
| **High-End Option** | N/A | **ViTPose (Server)** | Medical-grade accuracy, high cost. |
