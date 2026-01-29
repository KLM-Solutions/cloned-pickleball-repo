"""Main stroke classifier logic with velocity-aware segment detection."""
from typing import List, Dict, Any
from collections import Counter
from .heuristics import classify_frame, classify_stroke_enhanced


class StrokeClassifier:
    def __init__(self):
        self.history = []
        
    def classify_sequence(self, frames_metrics: List[Dict[str, Any]], target_type: str = None) -> Dict[str, Any]:
        """
        Classify a sequence of frames (a video clip) into a dominant stroke type.
        """
        if not frames_metrics:
            return {"type": "unknown", "confidence": 0.0}
            
        frame_votes = []
        history = []
        
        for metric in frames_metrics:
            # Use enhanced classification with history for velocity tracking
            stroke_type, conf, _ = classify_stroke_enhanced(metric, history, None, target_type=target_type)
            frame_votes.append(stroke_type)
            history.append(metric)
            # Keep history bounded
            if len(history) > 10:
                history = history[-10:]
            
        # Majority vote
        vote_counts = Counter(frame_votes)
        if not vote_counts:
            return {"type": "unknown", "confidence": 0.0}
            
        most_common, count = vote_counts.most_common(1)[0]
        confidence = count / len(frames_metrics)
        
        return {
            "type": most_common,
            "confidence": round(confidence, 2),
            "votes": dict(vote_counts)
        }

    def detect_segments(self, frames_metrics: List[Dict[str, Any]], target_type: str = None) -> List[Dict[str, Any]]:
        """
        Detect temporal segments where specific strokes occur.
        Uses velocity-aware classification with frame history.
        """
        segments = []
        if not frames_metrics:
            return segments

        current_type = None
        start_frame = 0
        conf_sum = 0.0
        conf_count = 0
        conf_max = 0.0
        history = []
        
        # Process each frame with history for velocity tracking
        for i, metric in enumerate(frames_metrics):
            # Use enhanced classification with history
            stroke_type, conf, sub_type = classify_stroke_enhanced(metric, history, None, target_type=target_type)
            frame_idx = int(metric.get("frame_idx", i))
            
            # Update history
            history.append(metric)
            if len(history) > 10:
                history = history[-10:]
            
            # If type changes, finalize previous segment
            if stroke_type != current_type:
                if current_type is not None:
                    prev_metric = frames_metrics[i - 1] if i - 1 >= 0 else metric
                    end_frame = int(prev_metric.get("frame_idx", frame_idx))
                    # End of segment
                    segments.append({
                        "start_frame": int(start_frame),
                        "end_frame": int(end_frame),
                        "stroke_type": current_type,
                        # Confidence: max per-frame confidence within the segment
                        "confidence": round(float(conf_max if conf_count > 0 else 0.0), 3),
                    })
                
                # Start new segment
                current_type = stroke_type
                start_frame = frame_idx
                conf_sum = float(conf)
                conf_count = 1
                conf_max = float(conf)
            else:
                # Accumulate confidence within the current segment
                conf_sum += float(conf)
                conf_count += 1
                conf_max = max(conf_max, float(conf))
        
        # Finalize last segment
        if current_type is not None:
            last_metric = frames_metrics[-1]
            end_frame = int(last_metric.get("frame_idx", len(frames_metrics) - 1))
            segments.append({
                "start_frame": int(start_frame),
                "end_frame": int(end_frame),
                "stroke_type": current_type,
                "confidence": round(float(conf_max if conf_count > 0 else 0.0), 3),
            })

        # Filter out very short segments (< 3 frames)
        # and None/unknown types
        valid_segments = [
            s for s in segments 
            if s["stroke_type"] is not None 
            and s["stroke_type"] != "unknown"
            and (s["end_frame"] - s["start_frame"] + 1) >= 3
        ]
        
        # MERGE nearby segments of the same type (gap <= 3 frames)
        merged = []
        for seg in valid_segments:
            if merged and merged[-1]["stroke_type"] == seg["stroke_type"]:
                gap = seg["start_frame"] - merged[-1]["end_frame"] - 1
                if gap <= 3:  # Merge if gap is small
                    merged[-1]["end_frame"] = seg["end_frame"]
                    merged[-1]["confidence"] = max(float(merged[-1].get("confidence", 0.0)), float(seg.get("confidence", 0.0)))
                    continue
            merged.append(seg.copy())
        
        return merged
