import traceback
import argparse
import json
import os
import cv2
import sys
import numpy as np
import time
import shutil
from pathlib import Path
import datetime
# FORCE FLUSH LOGGING
def log_debug(msg):
    ts = datetime.datetime.now().isoformat()
    print(f"[TRACK_PY_DEBUG {ts}] {msg}")
    sys.stdout.flush()
log_debug("Script loading...")
try:
    import torch
    print(f"CUDA Available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA Device Count: {torch.cuda.device_count()}")
        print(f"CUDA Device Name: {torch.cuda.get_device_name(0)}")
    else:
        print("WARNING: CUDA IS NOT AVAILABLE. TORCH RUNNING ON CPU.")
except ImportError:
    torch = None
    print("WARNING: Could not import torch explicitly.")
class NumpyEncoder(json.JSONEncoder):
    """ Custom encoder for numpy data types """
    def default(self, obj):
        if isinstance(obj, (np.intc, np.intp, np.int8,
                            np.int16, np.int32, np.int64, np.uint8,
                            np.uint16, np.uint32, np.uint64)):
            return int(obj)
        elif isinstance(obj, (np.float16, np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, (np.ndarray,)):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)
# Optional imports with graceful fallbacks
try:
    from boxmot import create_tracker  # type: ignore
except Exception:
    create_tracker = None  # fallback, no tracker
try:
    from ultralytics import YOLO  # type: ignore
except Exception:
    YOLO = None  # fallback to HOG detector
try:
    import mediapipe as mp  # type: ignore
    try:
        if not hasattr(mp, 'solutions'):
            raise ImportError("No solutions attribute")
    except Exception:
        mp = None
except Exception:
    mp = None
# NEW MODULAR IMPORTS - ENHANCED
# Ensure optional symbols are ALWAYS defined (avoid NameError in success-path imports).
classify_stroke_enhanced = None
try:
    from biomechanics import BiomechanicsAnalyzer, InjuryRiskDetector
    from biomechanics.angles import calculate_biomechanics_for_stroke
    from classification import StrokeClassifier
    # Optional: enhanced classifier lives in classification/heuristics.py
    try:
        from classification.heuristics import classify_stroke_enhanced  # type: ignore
    except Exception:
        classify_stroke_enhanced = None
except Exception as e:
    print(f"ERROR Importing Modules: {e}")
    BiomechanicsAnalyzer = None
    InjuryRiskDetector = None
    StrokeClassifier = None
    calculate_biomechanics_for_stroke = None
    classify_stroke_enhanced = None
def calculate_iou(box1, box2):
    """Calculate Intersection over Union (IoU) between two bounding boxes.
        Args:
        box1: [x1, y1, x2, y2] format
        box2: [x1, y1, x2, y2] format
        Returns:
        float: IoU value between 0 and 1
    """
    # Get intersection coordinates
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
        # No intersection
    if x2 <= x1 or y2 <= y1:
        return 0.0
        # Calculate intersection and union areas
    intersection = (x2 - x1) * (y2 - y1)
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = box1_area + box2_area - intersection
    return intersection / union if union > 0 else 0.0
def parse_args():
    parser = argparse.ArgumentParser(description="Run DeepOCSORT with ReID for Pickleball")
    parser.add_argument("--input_dir", type=str, required=True, help="Directory of input frame PNGs")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save annotated frames")
    parser.add_argument("--results_json", type=str, required=True, help="Path for output JSON")
    parser.add_argument("--yolo_model", type=str, default="yolov8n.pt", help="YOLOv8 weights")
    parser.add_argument("--reid_model", type=str, default="models/osnet_x1_0_msmt17.pt", help="Path to ReID weights")
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0",
        help="Device for YOLO inference. Use 'cuda:0' (default) on GPU workers, or 'cpu' for local CPU validation."
    )
    parser.add_argument("--target_point", type=str, default=None, help="Normalized click coordinates 'x,y'")
    parser.add_argument("--crop_region", type=str, default=None, help="Normalized crop region 'cx,cy,w,h'")
    parser.add_argument("--step", type=int, default=1, help="Process every Nth frame (frame skipping)")
    parser.add_argument("--stroke_type", type=str, default="serve", help="Hint for type of stroke being analyzed")
    parser.add_argument("--video_path", type=str, default=None, help="Path to the original video file for FPS detection")
    # Performance / long-video controls
    parser.add_argument(
        "--analysis_windows",
        type=str,
        default=None,
        help="Comma-separated time windows in seconds: 'start:end,start:end'. If set, metrics/pose are computed only inside windows."
    )
    parser.add_argument(
        "--process_only_windows",
        action="store_true",
        help="If set with --analysis_windows, only process frames that fall inside those windows."
    )
    parser.add_argument("--no_video_output", action="store_true", help="Skip writing annotated frame PNGs (results.json still written).")
    parser.add_argument("--no_skeleton_video", action="store_true", help="Skip writing skeleton_output.mp4 and drawing skeleton canvas.")
    parser.add_argument("--coarse_mode", action="store_true", help="Coarse scan mode (looser thresholds) for two-pass long-video analysis.")
    return parser.parse_args()
def get_video_fps(video_path):
    """Attempt to get FPS from video file."""
    if not video_path or not os.path.exists(video_path):
        return 30.0
    try:
        cap = cv2.VideoCapture(str(video_path))
        if cap.isOpened():
            fps = cap.get(cv2.CAP_PROP_FPS)
            cap.release()
            if fps > 0:
                return fps
    except:
        pass
    return 30.0
def main():
    args = parse_args()
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    results_json = Path(args.results_json)
    step = args.step
    output_dir.mkdir(parents=True, exist_ok=True)
    # 0. Get FPS for timing
    original_video = args.video_path
    fps = get_video_fps(original_video)
    log_debug(f"Initial FPS detected (baseline): {fps}")
    yolo_device = (args.device or "cuda:0").strip()

    # Parse analysis windows (in seconds)
    # Format: "start:end,start:end"
    windows_sec = []
    if args.analysis_windows:
        try:
            for part in str(args.analysis_windows).split(","):
                part = part.strip()
                if not part:
                    continue
                a, b = part.split(":")
                s = float(a)
                e = float(b)
                if e < s:
                    s, e = e, s
                windows_sec.append((max(0.0, s), max(0.0, e)))
        except Exception as e:
            print(f"Invalid --analysis_windows format: {args.analysis_windows} ({e})")
            windows_sec = []

    def _in_any_window(t: float) -> bool:
        if not windows_sec:
            return True
        for s, e in windows_sec:
            if s <= t <= e:
                return True
        return False
    # 1. Initialize detector (YOLO ONLY)
    model = None
    if YOLO is not None:
        try:
            model_path = args.yolo_model
            if not os.path.exists(model_path):
                alt = os.path.join("models", os.path.basename(model_path))
                if os.path.exists(alt):
                    model_path = alt
                # If neither exists, Ultralytics will auto-download, which is fine.
            print(f"Loading YOLO model: {model_path}")
            model = YOLO(model_path)
        except Exception as e:
            print(f"CRITICAL ERROR: Failed to load YOLO model: {e}")
            sys.exit(1) # Stop immediately if we can't load the modern model
    else:
        print("CRITICAL ERROR: Ultralytics YOLO library not installed.")
        sys.exit(1)
    # 2. Initialize tracker (BoxMOT DeepOCSORT) if available
    tracker = None
    BOXMOT_AVAILABLE = create_tracker is not None # Define BOXMOT_AVAILABLE based on create_tracker import
    if BOXMOT_AVAILABLE:
        try:
            tracker_config_path = Path(args.reid_model) # Use args.reid_model for consistency
            print(f"Loading DeepOCSORT via create_tracker: {tracker_config_path}")
            
            # KEY FIX: Use integer 0 for device, not 'cuda' string
            # The library requires specific device index for CUDA
            # per_class=False is important for general person tracking
            tracker_device = 0 if yolo_device.startswith("cuda") else "cpu"
            tracker = create_tracker(
                tracker_type='deepocsort',
                reid_weights=tracker_config_path, # Use the path from args
                device=tracker_device,
                half=bool(yolo_device.startswith("cuda")),
                per_class=False 
            )
            print("Tracker initialized successfully.")

            # --- PORTED TUNING FROM PICKLEBALL2 ---
            if tracker is not None:
                # FIXED: More conservative settings for stable tracking
                if hasattr(tracker, 'model') and hasattr(tracker.model, 'max_age'):
                    tracker.model.max_age = 300  # Keep ID for 10 seconds
                if hasattr(tracker, 'max_age'):
                    tracker.max_age = 300
                
                # FIXED: Require more consistent detections before assigning ID
                if hasattr(tracker, 'model') and hasattr(tracker.model, 'min_hits'):
                    tracker.model.min_hits = 1  # immediate ID
                if hasattr(tracker, 'min_hits'):
                    tracker.min_hits = 1
                    
                # Lower IOU threshold to prevent swaps
                if hasattr(tracker, 'model') and hasattr(tracker.model, 'iou_threshold'):
                    tracker.model.iou_threshold = 0.3  
                if hasattr(tracker, 'iou_threshold'):
                    tracker.iou_threshold = 0.3
            # --------------------------------------

        except Exception as e:
            print(f"Tracker init failed: {e}")
            tracker = None
    
    # Analyze CUDA environment (guard torch import)
    if torch is not None and torch.cuda.is_available():
        print(f"torch.cuda.is_available(): {torch.cuda.is_available()}")
        print(f"torch.cuda.device_count(): {torch.cuda.device_count()}")
        print(f"os.environ['CUDA_VISIBLE_DEVICES']: {os.environ.get('CUDA_VISIBLE_DEVICES', 'Not Set')}")

    if tracker is None:
        print("WARNING: Tracker unavailable. Proceeding without tracker.")
    # Initialize MediaPipe Pose (optional)
    pose = None
    mp_pose = None
    if mp is not None:
        try:
            if not hasattr(mp, 'solutions'):
                raise ImportError("MediaPipe module lookup failed: no 'solutions' attribute.")
            mp_pose = mp.solutions.pose
            pose = mp_pose.Pose(
                static_image_mode=False,
                model_complexity=1,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5,
            )
        except Exception as e:
            print(f"MediaPipe init failed: {e}. Skipping pose.")
            pose = None
            mp_pose = None
    # Initialize Analysis Modules - ENHANCED
    bio_analyzer = BiomechanicsAnalyzer() if BiomechanicsAnalyzer is not None else None
    classifier = StrokeClassifier() if StrokeClassifier is not None else None
    injury_detector = InjuryRiskDetector() if InjuryRiskDetector is not None else None
        # Update tracker settings for persistence (if tracker exists)
    if tracker is not None:
        # FIXED: More conservative settings for stable tracking
        if hasattr(tracker, 'model') and hasattr(tracker.model, 'max_age'):
            tracker.model.max_age = 300  # Keep ID for 10 seconds (at 30fps) - reduced from 40s
        if hasattr(tracker, 'max_age'):
            tracker.max_age = 300
                    # FIXED: Require more consistent detections before assigning ID
        if hasattr(tracker, 'model') and hasattr(tracker.model, 'min_hits'):
            tracker.model.min_hits = 1  # immediate ID
        if hasattr(tracker, 'min_hits'):
            tracker.min_hits = 1
                # Lower IOU threshold to prevent swaps
        if hasattr(tracker, 'model') and hasattr(tracker.model, 'iou_threshold'):
            tracker.model.iou_threshold = 0.3  
        if hasattr(tracker, 'iou_threshold'):
            tracker.iou_threshold = 0.3
    # Parse target point if provided
    target_point_norm = None
    crop_region_norm = None
    # Parse crop region
    if args.crop_region:
        try:
            # COORDINATE FIX: Parse as x1,y1,x2,y2 (not cx,cy,cw,ch)
            rx1, ry1, rx2, ry2 = map(float, args.crop_region.split(','))
            crop_region_norm = (rx1, ry1, rx2, ry2)
            log_debug(f"Received Crop Region (x1,y1,x2,y2): {crop_region_norm}")
        except:
            print("Invalid crop_region format")
    if args.target_point:
        try:
            px, py = map(float, args.target_point.split(','))
            target_point_norm = (px, py)
            print(f"Received target point (normalized): {target_point_norm}")
        except ValueError:
            print("Invalid target_point format. Expected 'x,y'. Using auto-selection.")
    # 3. Processing Loop
    all_files = sorted([p for p in input_dir.glob("*.png") if p.is_file()], key=lambda p: p.name)
    log_debug(f"Found {len(all_files)} total frames in input_dir.")

    def _frame_idx_from_path(p: Path) -> int:
        # Expected: frame_0001.png (1-based)
        try:
            return max(0, int(p.stem.split("_")[-1]) - 1)
        except Exception:
            return 0

    # If requested, process only frames inside the windows (coarse→refine pass-2 optimization)
    # IMPORTANT: Use frame index from filename so windowing doesn't break timestamps.
    if args.process_only_windows and windows_sec and len(all_files) > 0:
        filtered = []
        for p in all_files:
            fi = _frame_idx_from_path(p)
            if _in_any_window(fi / fps):
                filtered.append(p)
        all_files = filtered
        log_debug(f"Window-only processing enabled: {len(all_files)} frames selected from windows={windows_sec}")
    results = [] # Final frames to return to frontend
    all_frames_metrics = [] # To store metrics for sequence classification
    first_lock_frame = -1
    lock_wait_timeout = int(30 * 3) # Wait up to 3 seconds (assuming 30fps) for initial lock
    target_track_id = None
    lost_frames = 0 # Counter for frames where target is lost
    prev_wrist_px = None
    prev_time_sec = 0.0
    
    # NOTE: Single-ID invariant is enforced via POST-PROCESSING validation (see lines ~1070).
    # Real-time mid-stroke lock is not implemented since strokes are detected after all frames are processed.
    
    # Stats Tracking
    total_distance_m = 0.0
    # Keep these separate:
    # - prev_bbox_center: used for re-lock / nearest-person logic (must be bbox center)
    # - prev_bottom_center: used for distance traveled estimation (bottom of bbox)
    prev_bbox_center = None
    prev_bottom_center = None
        # --- Skeleton Video Init ---
    skeleton_dir = output_dir
    skeleton_dir.mkdir(parents=True, exist_ok=True)
        # Get frame size from first image
    if not all_files:
        print("Error: No frames found to process.")
        return
    first_frame_path = all_files[0]
    temp_img = cv2.imread(str(first_frame_path))
    skeleton_writer = None
    if temp_img is not None and (not args.no_skeleton_video) and (not args.no_video_output):
        h, w = temp_img.shape[:2]
        # Calculate resulting FPS after step
        # Base FPS is from original video (e.g. 30), result_fps is for output (e.g. 10)
        result_fps = fps / step
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        skeleton_out_path = skeleton_dir / "skeleton_output.mp4"
        skeleton_writer = cv2.VideoWriter(str(skeleton_out_path), fourcc, result_fps, (w, h))
        print(f"Skeleton video will be saved to: {skeleton_out_path} at {result_fps} FPS")
    else:
        if temp_img is None:
            print("Error: Could not read first frame.")
        else:
            print("Skeleton video disabled (no_skeleton_video or no_video_output).")
    print(f"Processing {len(all_files)} frames (Analysis every {step} steps)...")
    tracker_failed_count = 0
    saved_frame_count = 0
    for i, frame_path in enumerate(all_files):
        # Use original frame index derived from filename so windowing doesn't break timestamps.
        frame_idx = _frame_idx_from_path(frame_path)
        # We update the tracker EVERY frame for stability, but only analyze every 'step'
        is_analysis_frame = (frame_idx % step == 0)
        img = cv2.imread(str(frame_path))
        if img is None: continue
        height, width = img.shape[:2]
        skeleton_canvas = None if args.no_skeleton_video else np.zeros((height, width, 3), dtype=np.uint8)
        best_metrics = {} # Reset per frame
        landmarks_out = None  # Optional: MediaPipe landmarks for TS analyzeFrames fallback
        # A. Detection - HYBRID: YOLO only on analysis frames
        detections = np.empty((0, 6))
        if is_analysis_frame:
            # YOLO Inference ONLY on analysis frames to maintain speed
            try:
                # classes=[0] for person only, lower conf to 0.2
                yolo_preds = model.predict(img, classes=[0], conf=0.2, verbose=False, device=yolo_device)[0]
                if yolo_preds.boxes is not None:
                    ds = yolo_preds.boxes.data.cpu().numpy()
                    detections = ds if len(ds) > 0 else np.empty((0, 6))

                # STRICT CROP FILTERING: DISABLED
                # Reason: It filters out large foreground players if the crop is tight (low overlap ratio).
                # We want to track them and let the Selection Logic decide based on Crop Fill.
                # if crop_region_norm is not None and len(detections) > 0:
                #     cr_x1, cr_y1, cr_x2, cr_y2 = crop_region_norm
                #     c_px_x1, c_px_y1 = cr_x1 * width, cr_y1 * height
                #     c_px_x2, c_px_y2 = cr_x2 * width, cr_y2 * height
                #     
                #     valid_dets = []
                #     for det in detections:
                #         dx1, dy1, dx2, dy2 = det[:4]
                #         ix1 = max(dx1, c_px_x1)
                #         iy1 = max(dy1, c_px_y1)
                #         ix2 = min(dx2, c_px_x2)
                #         iy2 = min(dy2, c_px_y2)
                #         inter_area = max(0, ix2 - ix1) * max(0, iy2 - iy1)
                #         det_area = (dx2 - dx1) * (dy2 - dy1)
                #         overlap_ratio = inter_area / det_area if det_area > 0 else 0
                #         if overlap_ratio > 0.25:
                #             valid_dets.append(det)
                #     if len(valid_dets) < len(detections):
                #          log_debug(f"Frame {i}: Filtered {len(detections) - len(valid_dets)} detections outside crop.")
                #     detections = np.array(valid_dets) if len(valid_dets) > 0 else np.empty((0, 6))

            except Exception as e:
                print(f"YOLO inference failed: {e}")
        # Else: detections remains empty, tracker will use Kalman prediction
        tracks = []
        if tracker is not None:
            try:
                # Update tracker every frame to maintain ID stability
                # BoxMOT strictly expects (N, 6) for [x1, y1, x2, y2, conf, cls]
                if len(detections) == 0:
                    tracks = tracker.update(np.empty((0, 6)), img)
                else:
                    # Log shape for debugging
                    if i % 30 == 0:
                        log_debug(f"Frame {i}: Input detections shape: {detections.shape}")
                    tracks = tracker.update(detections, img)
                if i % 30 == 0:
                    log_debug(f"Frame {i}: Out tracks count: {len(tracks)}")
                    if len(tracks) > 0:
                        log_debug(f"Frame {i}: First track shape: {tracks[0].shape}")
            except Exception as e:
                tracker_failed_count += 1
                log_debug(f"ERROR: Tracker failed on frame {i}: {e}")
                if hasattr(detections, 'shape'):
                    log_debug(f"Detections shape was: {detections.shape}")
                tracks = []
        elif len(detections) > 0:
             # FALLBACK: If tracker failed to init, use raw YOLO detections as 'tracks'
             # Format: [x1, y1, x2, y2, id=-1, conf, cls]
             tracks = []
             for d in detections:
                 tracks.append([d[0], d[1], d[2], d[3], -1, d[4], d[5]])
             tracks = np.array(tracks)
             if i % 30 == 0:
                 log_debug(f"Frame {i}: Using RAW YOLO detections (Tracker unavailable)")
        best_box = None
        best_conf = 0.0
        best_metrics = {}
        found = False
        # C. Target Selection Logic - FIXED: Removed nested logic bug
        try:
            if len(tracks) > 0:
                # STICKY LOCK: Prioritize target_track_id if already locked
                if target_track_id is not None:
                    for t in tracks:
                        if int(t[4]) == int(target_track_id):
                            best_box = t[:4]
                            best_conf = t[5]
                            found = True
                            log_debug(f"  --> persistent lock on target ID {target_track_id}")
                            break
                                # If not found via persistent ID, look for a new match if we haven't locked yet
                if not found and target_track_id is None:
                    # FIRST TIME SELECTION (Initial target lock)
                    selected_id = -1
                                        # OPTION 1: ROI / Crop based selection (IoU) - PRIORITY METHOD
                    if crop_region_norm is not None:
                        # Convert normalized crop region to pixels
                        rx1, ry1, rx2, ry2 = crop_region_norm
                        # Ensure rcv values are derived from clean rx and ry
                        rcx = (rx1 + rx2) / 2
                        rcy = (ry1 + ry2) / 2
                                                # COORDINATE ALIGNMENT: Use rx1,ry1,rx2,ry2 standard
                        crop_box_px = [
                            rx1 * width,  # x1
                            ry1 * height, # y1
                            rx2 * width,  # x2
                            ry2 * height  # y2
                        ]
                        log_debug(f"Targeting logic: Frame size={width}x{height}")
                        log_debug(f"Targeting logic: Crop box (px) [{int(crop_box_px[0])}, {int(crop_box_px[1])}, {int(crop_box_px[2])}, {int(crop_box_px[3])}]")
                        best_score = 0
                        best_track_id = None
                        log_debug(f"Targeting logic: Crop center normalized ({rcx:.3f}, {rcy:.3f})")
                        for t in tracks:
                            x1, y1, x2, y2, tid, conf, cls = t[:7]
                            track_box = [x1, y1, x2, y2]
                            tcx, tcy = (x1 + x2) / 2, (y1 + y2) / 2
                            # Calculate IoU between crop region and track bbox
                            iou = calculate_iou(crop_box_px, track_box)
                            
                            # Calculate coverage (how much of person is in crop)
                            ix1, iy1 = max(x1, crop_box_px[0]), max(y1, crop_box_px[1])
                            ix2, iy2 = min(x2, crop_box_px[2]), min(y2, crop_box_px[3])
                            
                            coverage = 0
                            if ix2 > ix1 and iy2 > iy1:
                                inter_area = (ix2 - ix1) * (iy2 - iy1)
                                box_area = (x2 - x1) * (y2 - y1)
                                coverage = inter_area / box_area if box_area > 0 else 0
                                
                            # Distance from crop center (normalized)
                            dist_x = (tcx / width) - rcx
                            dist_y = (tcy / height) - rcy
                            dist_score = 1.0 - np.sqrt(dist_x**2 + dist_y**2)
                            
                            # Check if person center is INSIDE the crop box OR high coverage
                            is_inside = (crop_box_px[0] <= tcx <= crop_box_px[2] and 
                                         crop_box_px[1] <= tcy <= crop_box_px[3]) or coverage > 0.3
                            
                            # Final score: Favor Coverage and Center over raw IoU
                            score = (iou * 0.4) + (coverage * 1.0) + (dist_score * 0.6)
                            
                            if not is_inside:
                                score *= 0.1  # Penalty for being far outside the box
                            
                            log_debug(f"  Candidate Track {int(tid)}: center=({int(tcx)}, {int(tcy)}), IoU={iou:.3f}, Cov={coverage:.3f}, DistScore={dist_score:.3f}, Inside={is_inside}, Total={score:.3f}")

                            if score > best_score:
                                best_score = score
                                best_track_id = int(tid)
                        

                        if best_track_id is not None and best_score > 0.3:  # RELAXED threshold
                            selected_id = best_track_id
                            log_debug(f"--> CROP MATCH FOUND on frame {i}: Selected ID {selected_id} with score {best_score:.3f}")
                        
                        # LAST RESORT FALLBACK: If we have waited too long, just pick the best candidate
                        elif best_track_id is not None and i > (lock_wait_timeout // 2):
                             selected_id = best_track_id
                             log_debug(f"--> FORCE MATCH on frame {i}: Score {best_score:.3f} (Lower than threshold, but accepted as fallback)")

                        else:
                            # FALLBACK: If no track matches, try raw detections directly
                            if len(detections) > 0:
                                best_det_score = 0
                                for det_idx, det in enumerate(detections):
                                    dx1, dy1, dx2, dy2, dconf, dcls = det[:6]
                                    dtcx, dtcy = (dx1 + dx2) / 2, (dy1 + dy2) / 2
                                    # Use coverage and distance for raw detection matching
                                    dix1, diy1 = max(dx1, crop_box_px[0]), max(dy1, crop_box_px[1])
                                    dix2, diy2 = min(dx2, crop_box_px[2]), min(dy2, crop_box_px[3])
                                    d_coverage = 0
                                    if dix2 > dix1 and diy2 > diy1:
                                        d_inter_area = (dix2 - dix1) * (diy2 - diy1)
                                        d_box_area = (dx2 - dx1) * (dy2 - dy1)
                                        d_coverage = d_inter_area / d_box_area if d_box_area > 0 else 0
                                    
                                    d_score = (d_coverage * 1.5) # Prefer coverage
                                    if d_score > best_det_score:
                                        best_det_score = d_score
                                
                                if best_det_score > 0.5:
                                    log_debug(f"--> RAW DETECTION MATCH on frame {i} (Tracker returned 0 matches)")
                            
                            if i < lock_wait_timeout:
                                # Patiently wait for a good match in the first few seconds
                                if i % 10 == 0:
                                    log_debug(f"Waiting for crop match... (frame {i}/{lock_wait_timeout}, best score so far: {best_score:.3f})")
                                continue # Move to next frame without locking yet
                            else:
                                log_debug(f"Timeout waiting for crop match. Falling back to largest detection.")

                    # OPTION 2: Point-based Selection (Fallback)
                    elif target_point_norm is not None:
                        tx = target_point_norm[0] * width
                        ty = target_point_norm[1] * height
                        
                        min_dist = float('inf')
                        closest_id = None
                        
                        for t in tracks:
                            x1, y1, x2, y2, tid, conf, cls = t[:7]
                            if x1 <= tx <= x2 and y1 <= ty <= y2:
                                # Point is inside bbox - calculate distance to center
                                cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
                                dist = np.sqrt((cx - tx)**2 + (cy - ty)**2)
                                if dist < min_dist:
                                    min_dist = dist
                                    closest_id = int(tid)
                        
                        if closest_id is not None:
                            selected_id = closest_id
                            print(f"--> POINT MATCH: Selected ID {selected_id} at distance {min_dist:.1f}")

                    # OPTION 3: Largest Area (Auto-selection fallback)
                    # FIX: Strict Spatial + Size Filter
                    # 1. Must be large (> 30000 px)
                    # 2. Must be in NEAR COURT (Bottom 35% of screen, y2 > 0.65 * height)
                    if selected_id == -1:
                        max_area = 0
                        min_y_threshold = height * 0.65 # Bottom 35%
                        
                        for t in tracks:
                            x1, y1, x2, y2, tid, conf, cls = t[:7]
                            area = (x2 - x1) * (y2 - y1)
                            
                            # Filter: Large AND Low (Near Camera)
                            if area > 30000 and y2 > min_y_threshold:
                                if area > max_area:
                                    max_area = area
                                    selected_id = int(tid)
                        
                        if selected_id != -1:
                            print(f"--> AUTO MATCH: Selected Near-Court Target ID {selected_id} (Area: {int(max_area)})")

                    # OPTION 3: Largest Area (Auto-selection fallback)
                    # FIX: Strict Spatial + Size Filter
                    # 1. Must be large (> 30000 px)
                    # 2. Must be in NEAR COURT (Bottom 35% of screen, y2 > 0.65 * height)
                    if selected_id == -1:
                        max_area = 0
                        min_y_threshold = height * 0.65 # Bottom 35%
                        
                        for t in tracks:
                            x1, y1, x2, y2, tid, conf, cls = t[:7]
                            area = (x2 - x1) * (y2 - y1)
                            
                            # Filter: Large AND Low (Near Camera)
                            if area > 30000 and y2 > min_y_threshold:
                                if area > max_area:
                                    max_area = area
                                    selected_id = int(tid)
                        
                        if selected_id != -1:
                            print(f"--> AUTO MATCH: Selected Near-Court Target ID {selected_id} (Area: {int(max_area)})")
                    # Lock the target
                    if selected_id != -1:
                        target_track_id = selected_id
                        print(f"=== TARGET LOCKED: ID {target_track_id} ===")
                    else:
                        print("WARNING: No suitable target found")
                
                # Find our target in current tracks
                found = False
                for t in tracks:
                    x1, y1, x2, y2, tid, conf, cls = t[:7]
                    if int(tid) == target_track_id:
                        best_box = [float(x1), float(y1), float(x2), float(y2)]
                        best_conf = float(conf)
                        found = True
                        break
                                # If target is locked but not found in this frame, only continue if it's an analysis frame
                if not found and target_track_id is not None:
                    # STICKY LOCK: If we had a target ID, strictly look for IT first
                    for t in tracks:
                        if int(t[4]) == int(target_track_id):
                            best_box = t[:4]
                            best_conf = t[5]
                            found = True
                            log_debug(f"  --> TARGET ID {target_track_id} RE-LOCKED (Tracker Persistence)")
                            break

                if not found and len(detections) > 0:
                    # FALLBACK: If tracker lost ID, but we have YOLO detections in the ROI, take the best one
                    best_fallback_dist = 1e9
                    best_fallback_det = None
                    
                    for det in detections:
                        dx1, dy1, dx2, dy2, dconf, dcls = det[:6]
                        dtcx, dtcy = (dx1 + dx2) / 2, (dy1 + dy2) / 2
                        
                        # STRICT: Must be inside crop region if one was provided
                        is_in_crop = True
                        if crop_region_norm is not None:
                            rx1, ry1, rx2, ry2 = crop_region_norm
                            crop_x1, crop_y1 = rx1 * width, ry1 * height
                            crop_x2, crop_y2 = rx2 * width, ry2 * height
                            is_in_crop = (crop_x1 <= dtcx <= crop_x2 and crop_y1 <= dtcy <= crop_y2)
                        
                        if not is_in_crop:
                            continue  # Skip detections outside crop
                        
                        # Strict Distance Check (prev_bbox_center must exist if we are falling back)
                        if prev_bbox_center is not None:
                            dist = np.sqrt((dtcx - prev_bbox_center[0])**2 + (dtcy - prev_bbox_center[1])**2)
                            if dist < 40:  # STRICT LIMIT (40px)
                                if dist < best_fallback_dist:
                                    best_fallback_dist = dist
                                    best_fallback_det = det

                    if best_fallback_det is not None:
                        # Apply best fallback
                        dx1, dy1, dx2, dy2, dconf, dcls = best_fallback_det[:6]
                        best_box = [dx1, dy1, dx2, dy2]
                        best_conf = dconf
                        found = True
                        log_debug(f"  RE-LOCK FALLBACK (YOLO) on frame {i} (Dist: {best_fallback_dist:.1f}px, IN CROP)")
                
                # Update Persistence Logic
                if found:
                    lost_frames = 0
                    prev_bbox_center = (
                        (best_box[0] + best_box[2]) / 2,
                        (best_box[1] + best_box[3]) / 2
                    )
                else:
                    lost_frames += 1
                    # TIMEOUT: Keep tracking for 300 frames (10s at 30fps) - matches DeepOCSORT max_age
                    if lost_frames > 300 and target_track_id is not None:
                        log_debug(f"Target ID {target_track_id} lost for {lost_frames} frames (TIMEOUT). Keeping last known position.")
                        # DON'T reset target_track_id - keep trying to find the same person
                        # Only reset prev_center to force strict crop-based re-lock
                        lost_frames = 0

                if not found and target_track_id is not None and prev_bbox_center is not None:
                    # FIX: EDGE RE-ENTRY LOGIC
                    # If target is lost, check if a NEW track has appeared very close to the last known position.
                    # This handles the case where DeepOCSORT drops the ID (e.g. ID 1) and assigns a new one (ID 5) upon re-entry.
                    
                    px, py = prev_bbox_center
                    margin = 150
                    is_near_edge = (px < margin or px > width - margin or 
                                    py < margin or py > height - margin)
                    
                    # Search radius: Stricter than before to avoid swapping
                    # 100px normally, 150px if near edge (fast movement off-screen)
                    max_relock_dist = 150 if is_near_edge else 80 
                    
                    # Log waiting state periodically
                    if i % 30 == 0:
                        log_debug(f"Target ID {target_track_id} lost. Scanning for re-entry within {max_relock_dist}px (Edge: {is_near_edge})...")
                    
                    best_new_id = None
                    best_dist = float('inf')
                    
                    for t in tracks:
                        x1, y1, x2, y2, tid, conf, cls = t[:7]
                        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
                        
                        dist = np.sqrt((cx - px)**2 + (cy - py)**2)
                        
                        # Conditions for Re-Lock:
                        # 1. Distance valid
                        # 2. Not an unreasonable jump (prevents cross-court swaps)
                        if dist < max_relock_dist and dist < best_dist:
                            best_dist = dist
                            best_new_id = int(tid)
                            
                    if best_new_id is not None:
                        log_debug(f"--> RELOCK SUCCESS: Switched from lost ID {target_track_id} to new ID {best_new_id} (Dist: {best_dist:.1f}px)")
                        target_track_id = best_new_id
                        # Update found status immediately
                        for t in tracks:
                            if int(t[4]) == target_track_id:
                                x1, y1, x2, y2, tid, conf, cls = t[:7]
                                best_box = [float(x1), float(y1), float(x2), float(y2)]
                                best_conf = float(conf)
                                found = True
                                break
        except Exception as e:
            log_debug(f"CRITICAL ERROR in Target Selection: {e}")
            traceback.print_exc()

        if found:
            x1, y1, x2, y2 = best_box
            # Visual confirmation only for analysis frames
            if is_analysis_frame:
                cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                cv2.putText(img, f"TARGET ID:{target_track_id}", (int(x1), int(y1)-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
            # Distance Stats
            current_bottom_center = ((x1 + x2) / 2, y2)
            bbox_h = y2 - y1
            if prev_bottom_center is not None and bbox_h > 0:
                d_px = np.sqrt(
                    (current_bottom_center[0] - prev_bottom_center[0])**2 +
                    (current_bottom_center[1] - prev_bottom_center[1])**2
                )
                # Sanity check: if distance is too large, it might be a swap, but we allow it for re-lock
                scale = 1.75 / bbox_h # Assume 1.75m height
                total_distance_m += d_px * scale
            prev_bottom_center = current_bottom_center
            # Keep bbox-center updated for future re-lock logic (even if we skip analysis on some frames)
            prev_bbox_center = ((x1 + x2) / 2, (y1 + y2) / 2)

        # --- FIXED: Enhanced MediaPipe Pose Estimation on Crop ---
        # ONLY run for analysis frames, and only inside analysis windows (if provided)
        if is_analysis_frame and found and pose is not None and _in_any_window(frame_idx / fps):
                try:
                    # FIXED: Better padding calculation
                    pad = max(10, int(bbox_h * 0.15))  # At least 10px padding, 15% of height
                    y1_c, y2_c = max(0, int(y1)-pad), min(height, int(y2)+pad)
                    x1_c, x2_c = max(0, int(x1)-pad), min(width, int(x2)+pad)
                    
                    # FIXED: Ensure minimum crop size for pose detection
                    crop_width = x2_c - x1_c
                    crop_height = y2_c - y1_c
                    
                    if crop_width > 50 and crop_height > 80:  # Minimum size for pose detection
                        crop = img[y1_c:y2_c, x1_c:x2_c]
                        
                        # FIXED: Validate crop before processing
                        if crop.size > 0 and len(crop.shape) == 3:
                            crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
                            # Run pose inference
                            pose_results = pose.process(crop_rgb)
                        else:
                            pose_results = None
                    else:
                        pose_results = None

                    # FIXED: Process pose results if available
                    if pose_results is not None and pose_results.pose_landmarks:
                        # ... Logic continues ...
                        pass # Valid flow
                        crop_h, crop_w = crop.shape[:2]
                        print(f"DEBUG: Found pose landmarks, crop size: {crop_w}x{crop_h}")

                        # Export landmarks (MediaPipe order) for TS metrics fallback
                        try:
                            landmark_names = [
                                "nose",
                                "left_eye_inner", "left_eye", "left_eye_outer",
                                "right_eye_inner", "right_eye", "right_eye_outer",
                                "left_ear", "right_ear",
                                "mouth_left", "mouth_right",
                                "left_shoulder", "right_shoulder",
                                "left_elbow", "right_elbow",
                                "left_wrist", "right_wrist",
                                "left_pinky", "right_pinky",
                                "left_index", "right_index",
                                "left_thumb", "right_thumb",
                                "left_hip", "right_hip",
                                "left_knee", "right_knee",
                                "left_ankle", "right_ankle",
                                "left_heel", "right_heel",
                                "left_foot_index", "right_foot_index",
                            ]
                            landmarks_out = []
                            for li, lm in enumerate(pose_results.pose_landmarks.landmark):
                                landmarks_out.append({
                                    "name": landmark_names[li] if li < len(landmark_names) else "",
                                    "x": float(lm.x),
                                    "y": float(lm.y),
                                    "z": float(lm.z),
                                    "visibility": float(lm.visibility),
                                })
                        except Exception as e:
                            print(f"DEBUG: Failed to export landmarks: {e}")
                            landmarks_out = None
                                                # FIXED: Enhanced visualization with proper yellow/red colors
                        landmark_spec = mp.solutions.drawing_utils.DrawingSpec(
                            color=(0, 0, 255),     # RED joints
                            thickness=4,           # Slightly thicker for visibility
                            circle_radius=4        # Larger radius for better visibility
                        )
                        connection_spec = mp.solutions.drawing_utils.DrawingSpec(
                            color=(0, 255, 255),   # YELLOW connections 
                            thickness=3,           # Thicker lines for visibility
                            circle_radius=2        # Keep connection points smaller
                        )
                        # Filter connections to exclude head (indices 0-10)
                        filtered_connections = [
                            c for c in mp_pose.POSE_CONNECTIONS 
                            if c[0] > 10 and c[1] > 10
                        ]
                                                # Hide head landmarks
                        for idx in range(11): # 0 to 10
                            if idx < len(pose_results.pose_landmarks.landmark):
                                pose_results.pose_landmarks.landmark[idx].visibility = 0.0
                        # FIXED: Draw on Main Image 
                        try:
                            mp.solutions.drawing_utils.draw_landmarks(
                                img[y1_c:y2_c, x1_c:x2_c],
                                pose_results.pose_landmarks,
                                filtered_connections,
                                landmark_drawing_spec=landmark_spec,
                                connection_drawing_spec=connection_spec
                            )
                        except Exception as e:
                            print(f"DEBUG: Failed to draw pose on main image: {e}")
                        # FIXED: Draw on Skeleton Canvas (optional)
                        if skeleton_canvas is not None:
                            try:
                                mp.solutions.drawing_utils.draw_landmarks(
                                    skeleton_canvas[y1_c:y2_c, x1_c:x2_c],
                                    pose_results.pose_landmarks,
                                    filtered_connections,
                                    landmark_drawing_spec=landmark_spec,
                                    connection_drawing_spec=connection_spec
                                )
                            except Exception as e:
                                print(f"DEBUG: Failed to draw pose on skeleton canvas: {e}")
                        # --- ENHANCED BIOMECHANICS ANALYSIS ---
                        # OPTIMIZATION: Python only extracts Landmarks. TypeScript handles the Math.
                        
                        # Just pass basic structure for classifier
                        metrics = {} 
                        
                        # Still run classifier if needed for segmentation, but it might lack full metrics
                        # For now, we rely on TypeScript to backfill metrics
                        metrics["frame_idx"] = i
                        metrics["time_sec"] = round(i / fps, 3)
                        
                        best_metrics = metrics
                        all_frames_metrics.append(metrics)
                        
                        # Disabled Python-side Heavy Math:
                        # if bio_analyzer is not None:
                        #    bio_analyzer.update_landmarks(pose_results.pose_landmarks, crop_w, crop_h)
                        #    metrics = bio_analyzer.analyze_metrics(stroke_type=args.stroke_type)
                        #    ...
                            
                except Exception as e:
                    print(f"Pose/Biomech Error: {e}")
                    import traceback
                    traceback.print_exc()

        if not is_analysis_frame:
            continue
            
        # Save Frame Result - FIXED: Use sequential counter for out_filename
        saved_frame_count += 1
        out_filename = f"frame_{saved_frame_count:04d}.png" 
        time_sec = frame_idx / fps
        res_entry = {
            "frameIdx": int(saved_frame_count - 1),
            "frame_idx": int(frame_idx),
            "frameFilename": out_filename,
            "timestampSec": round(time_sec, 3),
            "bbox": best_box.tolist() if hasattr(best_box, 'tolist') else (best_box if best_box is not None else [0.0, 0.0, 0.0, 0.0]),
            "confidence": best_conf,
            "track_id": int(target_track_id) if target_track_id else -1,
            "metrics": best_metrics if isinstance(best_metrics, dict) else {},
            "landmarks": landmarks_out
        }
        results.append(res_entry)
        
        # Write Frame (optional)
        if not args.no_video_output:
            out_path = output_dir / out_filename
            cv2.imwrite(str(out_path), img)
            if skeleton_writer is not None and skeleton_canvas is not None:
                skeleton_writer.write(skeleton_canvas)

    log_debug(f"Final FPS used: {fps}")
    
    # --- POST-PROCESSING: STROKE CLASSIFICATION ---
    detected_strokes = []
    
    # Define classifier locally if global is missing/None
    classifier = StrokeClassifier() if StrokeClassifier else None
    
    if classifier is not None and len(all_frames_metrics) > 0:
        # Detect segments automatically with BIAS (guard signature/API mismatches)
        try:
            raw_segments = classifier.detect_segments(all_frames_metrics, target_type=args.stroke_type)
        except Exception as e:
            print(f"Classifier detect_segments failed (disabling classifier): {e}")
            classifier = None
            raw_segments = []
        
        # STRICT FILTERING: Only keep strokes matching the user's selection
        if args.stroke_type and args.stroke_type.lower() not in ['overall', 'none', '']:
            detected_strokes = []
            dropped_types = []
            for s in raw_segments:
                # Accept ONLY exact matches to requested stroke type
                if s['stroke_type'].lower() == args.stroke_type.lower():
                    
                    s['confidence'] = float(s['confidence']) # Ensure float
                    
                    # --- PEAK VELOCITY LOGIC (Ported from Manual Test) ---
                    s_start = s['start_frame']
                    s_end = s['end_frame']
                    max_v = 0.0
                    max_v_frame = s_start
                    
                    # Scan frames in this segment
                    # Note: all_frames_metrics indices might not align perfectly if frames were skipped
                    # But if we strictly appended, they should map via frame_idx
                    for m in all_frames_metrics:
                        f_idx = m.get('frame_idx', -1)
                        if s_start <= f_idx <= s_end:
                            v = m.get('wrist_velocity_mag', m.get('wrist_velocity_y', 0.0))
                            if v > max_v:
                                max_v = v
                                max_v_frame = f_idx
                    
                    s['peak_velocity'] = float(round(max_v, 2))
                    s['peak_frame_idx'] = int(max_v_frame)
                    s['peak_timestamp'] = float(round(max_v_frame / fps, 3))
                    
                    detected_strokes.append(s)
                    
                    # PRINT DETAILED TIMING LOG FOR USER (Verified Format)
                    start_sec = round(s['start_frame'] / fps, 2)
                    end_sec = round(s['end_frame'] / fps, 2)
                    print(f"[RESULT] Found '{s['stroke_type']}': Frame {s['start_frame']}-{s['end_frame']} (t={start_sec}s to {end_sec}s)")
                    print(f"       -> PEAK: Frame {s['peak_frame_idx']} (t={s['peak_timestamp']}s, V={s['peak_velocity']})")
                else:
                    dropped_types.append(s['stroke_type'])
            
            print(f"Strict Filter: Kept {len(detected_strokes)} segments matching '{args.stroke_type}'")
            if dropped_types:
                print(f"  (Dropped {len(dropped_types)} others: {dropped_types})")
        else:
            detected_strokes = raw_segments

        # Per-stroke filtering + cooldown (multi-stroke for long videos)
        stroke_key = (args.stroke_type or "").lower()
        cfg_by_type = {
            # NOTE: peak_velocity is based on wrist_velocity_mag (normalized coords/sec).
            "serve": {"min_conf": 0.78, "min_peak_v": 0.18, "min_len_frames": 4, "cooldown_sec": 1.0},
            "groundstroke": {"min_conf": 0.78, "min_peak_v": 0.16, "min_len_frames": 4, "cooldown_sec": 0.8},
            "drive": {"min_conf": 0.78, "min_peak_v": 0.16, "min_len_frames": 4, "cooldown_sec": 0.8},
            "volley": {"min_conf": 0.75, "min_peak_v": 0.10, "min_len_frames": 3, "cooldown_sec": 0.6},
            "dink": {"min_conf": 0.72, "min_peak_v": 0.06, "min_len_frames": 3, "cooldown_sec": 0.6},
            "overhead": {"min_conf": 0.80, "min_peak_v": 0.14, "min_len_frames": 3, "cooldown_sec": 1.0},
        }
        cfg = cfg_by_type.get(stroke_key, {"min_conf": 0.75, "min_peak_v": 0.10, "min_len_frames": 3, "cooldown_sec": 0.8})

        if args.coarse_mode:
            # Coarse pass: more lenient to find candidate windows.
            cfg = {
                **cfg,
                "min_conf": max(0.55, float(cfg["min_conf"]) - 0.18),
                "min_peak_v": max(0.0, float(cfg["min_peak_v"]) * 0.5),
                "cooldown_sec": 0.0,
            }

        cooldown_frames = int(float(cfg.get("cooldown_sec", 0.0)) * float(fps))
        filtered = []
        for s in sorted(detected_strokes, key=lambda x: int(x.get("start_frame", 0))):
            try:
                start_f = int(s.get("start_frame", 0))
                end_f = int(s.get("end_frame", start_f))
                seg_len = (end_f - start_f + 1)
                peak_v = float(s.get("peak_velocity", 0.0) or 0.0)
                conf = float(s.get("confidence", 0.0) or 0.0)
            except Exception:
                continue

            if seg_len < int(cfg.get("min_len_frames", 3)):
                continue
            if conf < float(cfg.get("min_conf", 0.75)):
                continue
            if peak_v < float(cfg.get("min_peak_v", 0.0)):
                continue

            # Cooldown: prevent duplicates around the same physical hit
            if cooldown_frames > 0 and filtered:
                prev_end = int(filtered[-1].get("end_frame", filtered[-1].get("start_frame", 0)))
                if start_f <= prev_end + cooldown_frames:
                    continue

            filtered.append(s)

        detected_strokes = filtered
        
        # ============================================================
        # SINGLE-ID INVARIANT: Validate and attach track_id to strokes
        # ============================================================
        # Each stroke MUST be associated with exactly one track_id.
        # If frames in a stroke window have mixed IDs, we flag it.
        validated_strokes = []
        for s in detected_strokes:
            start_f = int(s.get("start_frame", 0))
            end_f = int(s.get("end_frame", start_f))
            
            # Collect track_ids from all frames in this stroke window
            stroke_track_ids = set()
            for res in results:
                res_frame_idx = res.get("frame_idx", res.get("frameIdx", -1))
                if start_f <= res_frame_idx <= end_f:
                    tid = res.get("track_id", -1)
                    if tid != -1:
                        stroke_track_ids.add(tid)
            
            # SINGLE-ID INVARIANT CHECK
            if len(stroke_track_ids) == 0:
                # No valid track_id found; use target_track_id as fallback
                s["track_id"] = int(target_track_id) if target_track_id else -1
                validated_strokes.append(s)
            elif len(stroke_track_ids) == 1:
                # ✓ VALID: Single ID throughout stroke
                s["track_id"] = int(stroke_track_ids.pop())
                validated_strokes.append(s)
            else:
                # ⚠ VIOLATION: Multiple IDs in one stroke window
                print(f"[WARNING] Single-ID violation in stroke {s.get('stroke_type')} "
                      f"frames {start_f}-{end_f}: IDs={stroke_track_ids}")
                # Use the DOMINANT track_id (most occurrences)
                id_counts = {}
                for res in results:
                    res_frame_idx = res.get("frame_idx", res.get("frameIdx", -1))
                    if start_f <= res_frame_idx <= end_f:
                        tid = res.get("track_id", -1)
                        if tid != -1:
                            id_counts[tid] = id_counts.get(tid, 0) + 1
                
                if id_counts:
                    dominant_id = max(id_counts.keys(), key=lambda k: id_counts[k])
                    s["track_id"] = int(dominant_id)
                    s["multi_id_warning"] = True  # Flag for debugging
                    validated_strokes.append(s)
                    print(f"         -> Using dominant ID: {dominant_id} (counts: {id_counts})")
        
        detected_strokes = validated_strokes
        print(f"[SINGLE-ID] Validated {len(detected_strokes)} strokes")
        print(f"Final Output strokes: {[s.get('stroke_type') for s in detected_strokes]}")
    else:
        # Fallback if classifier failed or no metrics
        print("Classifier unavailable or no metrics collected. Using hint.")
        detected_strokes = [{
            "stroke_type": args.stroke_type, # User hint
            "start_frame": 0,
            "startSec": 0.0,
            "confidence": 1.0
        }]

    # Calculate time seconds for frontend
    # Use the original video FPS as reference for timestamps
    for s in detected_strokes:
        # Ensure startSec/endSec exist (camelCase for TS)
        s_start = s.get("start_frame", 0)
        s_end = s.get("end_frame", 0)
        s["startSec"] = round(s_start / fps, 2)
        s["endSec"] = round(s_end / fps, 2)
        # Add snake_case too just in case
        s["start_sec"] = s["startSec"]
        s["end_sec"] = s["endSec"]
        # Ensure type is present
        if "type" not in s and "stroke_type" in s:
            s["type"] = s["stroke_type"]

    # Debug logging: stroke timing summary for RunPod logs
    try:
        for s in detected_strokes:
            stype = s.get("stroke_type", s.get("type", "unknown"))
            start_sec = s.get("startSec", 0.0)
            end_sec = s.get("endSec", start_sec)
            peak_ts = s.get("peak_timestamp", None)
            peak_v = s.get("peak_velocity", None)
            if peak_ts is not None and peak_v is not None:
                print(f"[STROKE] {stype} | {start_sec:.2f}s → {end_sec:.2f}s | peak @ {float(peak_ts):.2f}s | v={float(peak_v):.2f}")
            else:
                print(f"[STROKE] {stype} | {start_sec:.2f}s → {end_sec:.2f}s")
    except Exception as e:
        print(f"[STROKE] logging failed: {e}")
    # ENHANCED: Get injury risk summary
    injury_risk_summary = {}
    if injury_detector is not None:
        try:
            injury_risk_summary = injury_detector.get_session_summary()
            print(f"Injury Risk Summary: {injury_risk_summary.get('overall_risk', 'unknown')} risk")
            if injury_risk_summary.get('alerts'):
                print(f"  Alerts: {len(injury_risk_summary['alerts'])} detected")
        except Exception as e:
            print(f"Failed to generate injury risk summary: {e}")
        # Save JSON with ENHANCED schema
    final_output = {
        "frames": results, # Per-frame data matched to frontend 'frames' key
        "strokes": detected_strokes, # Segments
        "summary": {
            "total_distance_m": round(total_distance_m, 2),
            "avg_speed_kmh": round((total_distance_m / (len(all_files) / fps) * 3.6), 2) if len(all_files) > 0 else 0,
            "tracked_duration_sec": round(len(all_files) / fps, 1),
            "dominant_stroke": detected_strokes[0]["stroke_type"] if detected_strokes else "unknown"
        },
        "injury_risk_summary": injury_risk_summary  # NEW: Injury risk analysis
    }
    
    with open(results_json, "w") as f:
        json.dump(final_output, f, indent=2, cls=NumpyEncoder)
    
    print("Tracking complete.")
    
    if skeleton_writer is not None:
        skeleton_writer.release()

if __name__ == "__main__":
    main()