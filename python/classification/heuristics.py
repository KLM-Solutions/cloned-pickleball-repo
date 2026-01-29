"""
Enhanced stroke classification with research-based heuristics.
Based on comprehensive pickleball biomechanics research.

Key insight: Strokes are distinguished by MOTION (velocity) not just POSITION.
"""
from typing import Tuple, Dict, Any, Optional, List
import math

# ============================================================================
# RESEARCH-BASED THRESHOLDS
# ============================================================================
# Based on pickleball biomechanics:
# - Serve: Underhand, wrist below waist, UPWARD swing with HIGH velocity
# - Groundstroke: Horizontal swing, wrist at waist, strong hip rotation
# - Volley: Compact punch, minimal backswing, at kitchen line
# - Overhead: Wrist above head, high shoulder abduction, downward swing
# - Dink: Soft touch, minimal swing, low body position
# ============================================================================

THRESHOLDS = {
    'overhead': {
        'min_shoulder_abduction': 110,      # High arm position
        'min_wrist_above_nose': 0.05,       # Wrist clearly above head (normalized)
        'confidence_high': 130              # Very high shoulder for max confidence
    },
    'serve': {
        # Position thresholds
        'max_wrist_above_hip': 0.20,        # Wrist at/below waist (generous tolerance)
        'min_shoulder_abduction': 20,       # Some arm extension
        'max_shoulder_abduction': 100,      # Not overhead
        'min_hip_rotation': 3,              # Minimal body rotation required
        # Velocity thresholds - LOWERED for normalized coords (0-1 space)
        'min_wrist_velocity': 0.008,        # Normalized velocity per frame
        'peak_velocity_threshold': 0.02,    # Peak velocity for strong serve
        # Temporal thresholds
        'min_upward_frames': 2,             # Need sustained upward motion
    },
    'groundstroke': {
        'min_shoulder_abduction': 45,       # Moderate arm extension
        'max_shoulder_abduction': 110,      # Below overhead
        'min_hip_rotation': 10,             # Good rotation for power
        'min_horizontal_velocity': 0.01,    # Side-to-side motion
    },
    'volley': {
        'min_shoulder_abduction': 30,
        'max_shoulder_abduction': 85,
        'max_wrist_velocity': 0.015,        # Compact motion
        'punch_min_shoulder': 55,           # Punch volley threshold
    },
    'dink': {
        'max_shoulder_abduction': 55,       # Low arm position
        'max_wrist_velocity': 0.012,        # Very soft touch
        'min_knee_flexion': 15,             # Some knee bend
        'max_knee_flexion': 70,             # Not too low
    }
}


def compute_velocity(current: Dict, history: List[Dict], key_y: str, key_x: str = None) -> Tuple[float, float, float]:
    """
    Compute velocity from frame history.
    Returns: (velocity_y, velocity_x, velocity_magnitude)
    
    velocity_y < 0 means moving UP (y increases downward in image coords)
    """
    if len(history) < 1:
        return (0.0, 0.0, 0.0)
    
    curr_y = current.get(key_y, 0.5)
    prev_y = history[-1].get(key_y, curr_y)
    
    vel_y = curr_y - prev_y  # negative = moving up
    
    vel_x = 0.0
    if key_x:
        curr_x = current.get(key_x, 0.5)
        prev_x = history[-1].get(key_x, curr_x)
        vel_x = curr_x - prev_x
    
    magnitude = math.sqrt(vel_y**2 + vel_x**2)
    
    return (vel_y, vel_x, magnitude)


def count_upward_frames(history: List[Dict], key: str = 'right_wrist_y', min_frames: int = 3) -> int:
    """Count consecutive frames with upward wrist motion at end of history."""
    if len(history) < 2:
        return 0
    
    count = 0
    for i in range(len(history) - 1, 0, -1):
        curr_y = history[i].get(key, 0.5)
        prev_y = history[i-1].get(key, 0.5)
        if curr_y < prev_y:  # Moving up
            count += 1
        else:
            break
    return count


# ============================================================================
# STROKE DETECTION FUNCTIONS
# ============================================================================

def is_overhead(metrics: Dict[str, Any], history: List[Dict] = None) -> Tuple[bool, float]:
    """
    Detect overhead smash.
    
    Key criteria:
    - Wrist ABOVE head (nose)
    - High shoulder abduction (> 110°)
    - Downward velocity at contact
    """
    wrist_y = metrics.get('right_wrist_y', 1.0)
    nose_y = metrics.get('nose_y', 0.5)
    shoulder_abd = metrics.get('right_shoulder_abduction', 0)
    
    # Wrist above head check (y decreases upward)
    wrist_above_head = (nose_y - wrist_y) >= THRESHOLDS['overhead']['min_wrist_above_nose']
    
    # High shoulder check
    high_shoulder = shoulder_abd >= THRESHOLDS['overhead']['min_shoulder_abduction']
    
    if wrist_above_head and high_shoulder:
        confidence = 0.85
        if shoulder_abd >= THRESHOLDS['overhead']['confidence_high']:
            confidence = 0.95
        return (True, confidence)
    
    return (False, 0.0)


def is_serve(metrics: Dict[str, Any], history: List[Dict]) -> Tuple[bool, float]:
    """
    Detect serve based on VELOCITY + POSITION.
    
    Key criteria (ALL required for high confidence):
    1. Wrist at or below waist level
    2. UPWARD wrist velocity (the swing)
    3. Moderate shoulder abduction (20-100°)
    4. Hip rotation present (> 3°)
    
    The key differentiator: UPWARD VELOCITY at contact point.
    Also accounts for follow-through phase (wrist going UP after contact).
    """
    wrist_y = metrics.get('right_wrist_y', 0.5)
    hip_y = metrics.get('right_hip_y', 0.5)
    shoulder_abd = metrics.get('right_shoulder_abduction', 0)
    hip_rotation = abs(metrics.get('hip_rotation_deg', 0))
    
    # 1. Position check: wrist at/below waist (with generous tolerance for follow-through)
    wrist_height_diff = hip_y - wrist_y  # positive = wrist above hip
    wrist_at_waist = wrist_height_diff <= THRESHOLDS['serve']['max_wrist_above_hip']
    # Also allow wrist ABOVE waist during follow-through (wrist rising after contact)
    wrist_during_followthrough = wrist_y < hip_y  # Wrist is above waist
    
    # 2. Shoulder in serve range (not overhead, not too low)
    shoulder_in_range = (THRESHOLDS['serve']['min_shoulder_abduction'] <= 
                         shoulder_abd <= 
                         THRESHOLDS['serve']['max_shoulder_abduction'])
    
    # 3. Hip rotation check
    has_hip_rotation = hip_rotation >= THRESHOLDS['serve']['min_hip_rotation']
    
    # 4. VELOCITY check (CRITICAL)
    vel_y, _, vel_mag = compute_velocity(metrics, history, 'right_wrist_y')
    is_moving_up = vel_y < -0.005  # Moving upward (negative y = up)
    is_moving_down = vel_y > 0.005  # Moving downward (follow-through)
    has_velocity = vel_mag >= THRESHOLDS['serve']['min_wrist_velocity']
    has_strong_velocity = vel_mag >= THRESHOLDS['serve']['peak_velocity_threshold']
    
    # 5. Check if RECENT history had a serve-like peak (sticky detection)
    recent_had_serve_velocity = False
    if len(history) >= 2:
        # Check last 5 frames for high velocity
        for i in range(max(0, len(history) - 5), len(history)):
            if i > 0:
                prev = history[i]
                prev_wrist = prev.get('right_wrist_y', 0.5)
                earlier_wrist = history[i-1].get('right_wrist_y', prev_wrist)
                prev_vel_mag = abs(prev_wrist - earlier_wrist)
                if prev_vel_mag >= 0.015:  # Had significant velocity recently
                    recent_had_serve_velocity = True
                    break
    
    # 6. Sustained upward motion check
    upward_frames = count_upward_frames(history + [metrics], 'right_wrist_y')
    sustained_upward = upward_frames >= THRESHOLDS['serve']['min_upward_frames']
    
    # Scoring: Need multiple criteria to pass
    score = 0
    if wrist_at_waist:
        score += 1
    if shoulder_in_range:
        score += 1
    if has_hip_rotation:
        score += 1
    if is_moving_up:
        score += 2  # Double weight for upward motion (the key serve signature)
    if has_velocity:
        score += 1
    if sustained_upward:
        score += 1
    if has_strong_velocity:
        score += 2  # Double weight for strong velocity
    
    # STICKY: If recently had serve velocity AND wrist is in follow-through position
    if recent_had_serve_velocity and wrist_during_followthrough and shoulder_in_range:
        score += 2  # Boost for follow-through phase
    
    # Need at least 3 points to classify as serve
    if score >= 3:
        confidence = min(0.55 + (score - 3) * 0.06, 0.95)
        return (True, confidence)
    
    return (False, 0.0)


def is_groundstroke(metrics: Dict[str, Any], history: List[Dict] = None) -> Tuple[bool, float, str]:
    """
    Detect groundstroke (forehand/backhand drive).
    
    Key criteria:
    - Moderate shoulder abduction (45-110°)
    - Hip rotation present (> 15°)
    - Horizontal swing path
    """
    shoulder_abd = metrics.get('right_shoulder_abduction', 0)
    hip_rotation = abs(metrics.get('hip_rotation_deg', 0))
    
    # Shoulder range check
    shoulder_in_range = (THRESHOLDS['groundstroke']['min_shoulder_abduction'] <= 
                         shoulder_abd <= 
                         THRESHOLDS['groundstroke']['max_shoulder_abduction'])
    
    if not shoulder_in_range:
        return (False, 0.0, '')
    
    # Hip rotation check
    has_hip_rotation = hip_rotation >= THRESHOLDS['groundstroke']['min_hip_rotation']
    
    # Determine forehand vs backhand
    r_shoulder_x = metrics.get('right_shoulder_x', 0.5)
    l_shoulder_x = metrics.get('left_shoulder_x', 0.5)
    is_forehand = r_shoulder_x > l_shoulder_x
    
    confidence = 0.75
    if has_hip_rotation:
        confidence += 0.10
    
    sub_type = 'forehand' if is_forehand else 'backhand'
    if shoulder_abd >= 70:
        sub_type += '_drive'
    else:
        sub_type += '_control'
    
    return (True, min(confidence, 0.95), sub_type)


def is_volley(metrics: Dict[str, Any], history: List[Dict] = None, position_data: Optional[Dict] = None) -> Tuple[bool, float, str]:
    """
    Detect volley (punch or block).
    
    Key criteria:
    - Moderate shoulder abduction (30-80°)
    - Compact motion (low velocity)
    - At kitchen line (if position available)
    """
    shoulder_abd = metrics.get('right_shoulder_abduction', 0)
    
    # Shoulder range check
    shoulder_in_range = (THRESHOLDS['volley']['min_shoulder_abduction'] <= 
                         shoulder_abd <= 
                         THRESHOLDS['volley']['max_shoulder_abduction'])
    
    if not shoulder_in_range:
        return (False, 0.0, '')
    
    # Velocity check (volleys are compact)
    _, _, vel_mag = compute_velocity(metrics, history or [], 'right_wrist_y')
    is_compact = vel_mag <= THRESHOLDS['volley']['max_wrist_velocity']
    
    if not is_compact and history:
        return (False, 0.0, '')
    
    # Sub-type
    if shoulder_abd >= THRESHOLDS['volley']['punch_min_shoulder']:
        sub_type = 'punch'
        confidence = 0.85
    else:
        sub_type = 'block'
        confidence = 0.80
    
    # Boost for kitchen line
    if position_data and position_data.get('at_kitchen_line'):
        confidence += 0.05
    
    return (True, min(confidence, 0.95), sub_type)


def is_dink(metrics: Dict[str, Any], history: List[Dict]) -> Tuple[bool, float, str]:
    """
    Detect dink shot.
    
    Key criteria:
    - Low shoulder abduction (< 50°)
    - Very low velocity (soft touch)
    - Low body position (knee flexion)
    - Wrist below waist
    """
    shoulder_abd = metrics.get('right_shoulder_abduction', 0)
    knee_flexion = metrics.get('right_knee_flexion', 180)
    wrist_y = metrics.get('right_wrist_y', 0.5)
    hip_y = metrics.get('right_hip_y', 0.5)
    
    # Low shoulder check
    low_shoulder = shoulder_abd <= THRESHOLDS['dink']['max_shoulder_abduction']
    
    # Velocity check (soft touch)
    _, _, vel_mag = compute_velocity(metrics, history, 'right_wrist_y')
    soft_touch = vel_mag <= THRESHOLDS['dink']['max_wrist_velocity']
    
    # Low body check
    low_body = (THRESHOLDS['dink']['min_knee_flexion'] <= 
                knee_flexion <= 
                THRESHOLDS['dink']['max_knee_flexion'])
    
    # Contact point check
    wrist_below_waist = wrist_y >= hip_y
    
    # Need multiple criteria
    score = 0
    if low_shoulder:
        score += 1
    if soft_touch:
        score += 2  # Double weight
    if low_body:
        score += 1
    if wrist_below_waist:
        score += 1
    
    if score >= 3:
        confidence = min(0.70 + (score - 3) * 0.08, 0.95)
        return (True, confidence, 'soft_game')
    
    return (False, 0.0, '')


# ============================================================================
# MAIN CLASSIFICATION FUNCTION
# ============================================================================

def classify_stroke_enhanced(
    metrics: Dict[str, Any],
    frame_history: List[Dict],
    position_data: Optional[Dict] = None,
    target_type: Optional[str] = None
) -> Tuple[str, float, str]:
    """
    Enhanced stroke classification using velocity + position heuristics.
    
    Priority order (most distinctive first):
    1. Overhead (wrist above head)
    2. Serve (upward velocity + waist contact)
    3. Dink (soft touch + low position)
    4. Volley (compact punch)
    5. Groundstroke (default for baseline shots)
    """
    
    # 1. OVERHEAD - Most distinctive
    is_oh, oh_conf = is_overhead(metrics, frame_history)
    if is_oh and oh_conf >= 0.80:
        return ('overhead', oh_conf, 'smash')
    
    # 2. SERVE - Check with velocity
    is_srv, srv_conf = is_serve(metrics, frame_history)
    
    if target_type == 'serve':
        # Serve videos must be STRICT to avoid flagging "walking/ready stance" as serves.
        # Only emit 'serve' when serve criteria are actually met.
        if is_srv and srv_conf >= 0.75:
            return ('serve', max(srv_conf, 0.85), 'underhand')
    else:
        # Strict mode
        if is_srv and srv_conf >= 0.75:
            return ('serve', srv_conf, 'underhand')
    
    # 3. DINK - Low, soft shots (SKIP if target is serve)
    if target_type != 'serve':
        is_dnk, dnk_conf, dnk_sub = is_dink(metrics, frame_history)
        
        if target_type == 'dink':
            if is_dnk and dnk_conf >= 0.70:
                return ('dink', max(dnk_conf, 0.80), dnk_sub)
        else:
            if is_dnk and dnk_conf >= 0.75:
                return ('dink', dnk_conf, dnk_sub)
    
    # 4. VOLLEY - Compact punch (SKIP if target is serve)
    if target_type != 'serve':
        is_vol, vol_conf, vol_sub = is_volley(metrics, frame_history, position_data)
        if is_vol and vol_conf >= 0.75:
            return ('volley', vol_conf, vol_sub)
    
    # 5. GROUNDSTROKE - Default for baseline (SKIP if target is serve)
    if target_type != 'serve':
        is_gs, gs_conf, gs_sub = is_groundstroke(metrics, frame_history)
        
        if target_type in ['groundstroke', 'drive']:
            if is_gs:
                return ('groundstroke', max(gs_conf, 0.80), gs_sub)
        else:
            if is_gs:
                return ('groundstroke', gs_conf, gs_sub)
    
    # Default fallback
    return ('unknown', 0.40, 'unclassified')


def classify_frame(metrics: Dict[str, float], landmarks: Any, target_type: Optional[str] = None) -> Optional[str]:
    """
    Legacy compatibility function.
    Single frame classification (no history).
    
    Returns: stroke_type string or None if unclassified
    """
    stroke_type, conf, _ = classify_stroke_enhanced(metrics, [], None, target_type=target_type)
    
    # Only return if confidence is decent
    if conf >= 0.55:
        return stroke_type
    return None
