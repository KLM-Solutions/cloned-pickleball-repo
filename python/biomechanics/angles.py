"""
Refined biomechanics angle calculations with validated thresholds.
Based on comprehensive pickleball biomechanics research.
"""
import numpy as np
from typing import Tuple, Dict, Any

# Research-based safe angle ranges
ANGLE_THRESHOLDS = {
    'shoulder_abduction': {
        'safe': (0, 120),
        'caution': (120, 140),
        'high_risk': (140, 160),
        'critical': (160, 180),
        'overhead_exception': (120, 160)  # Acceptable for overhead smash
    },
    'elbow_flexion': {
        'serve': (90, 120),
        'groundstroke': (120, 160),
        'dink': (90, 110),
        'volley': (90, 120),
        'overhead': (90, 170)
    },
    'knee_flexion': {
        'ready': (20, 30),
        'dinking': (30, 45),
        'serving': (15, 25),
        'overhead': (30, 45),
        'safe_max': 90  # Above this = high stress
    },
    'hip_rotation': {
        'serve': (10, 20),
        'groundstroke': (45, 90),
        'dink': (0, 15),
        'volley': (10, 30),
        'overhead': (60, 120),
        'power_threshold': 30  # Below this = poor power generation
    }
}


def calculate_angle(a, b, c) -> float:
    """
    Calculate angle at point b formed by points a-b-c.
    
    Args:
        a, b, c: MediaPipe landmark objects with x, y, z attributes
    
    Returns:
        Angle in degrees (0-180)
    """
    if not all([a, b, c]):
        return 0.0
    
    # Convert to numpy arrays
    a_pos = np.array([a.x, a.y, a.z]) if hasattr(a, 'z') else np.array([a.x, a.y, 0])
    b_pos = np.array([b.x, b.y, b.z]) if hasattr(b, 'z') else np.array([b.x, b.y, 0])
    c_pos = np.array([c.x, c.y, c.z]) if hasattr(c, 'z') else np.array([c.x, c.y, 0])
    
    # Calculate vectors
    ba = a_pos - b_pos
    bc = c_pos - b_pos
    
    # Calculate angle using dot product
    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-6)
    cosine_angle = np.clip(cosine_angle, -1.0, 1.0)  # Avoid numerical errors
    
    angle = np.degrees(np.arccos(cosine_angle))
    
    return float(angle)


def calculate_shoulder_abduction_validated(shoulder, elbow, hip, stroke_type: str = 'groundstroke') -> Dict[str, Any]:
    """
    Calculate shoulder abduction with validation against safe ranges.
    
    Args:
        shoulder, elbow, hip: MediaPipe landmarks
        stroke_type: Type of stroke being performed
    
    Returns:
        Dictionary with angle, risk_level, and validation info
    """
    angle = calculate_angle(hip, shoulder, elbow)
    
    thresholds = ANGLE_THRESHOLDS['shoulder_abduction']
    
    # Determine risk level
    if angle < thresholds['safe'][1]:
        risk = 'safe'
        risk_color = 'green'
    elif angle < thresholds['caution'][1]:
        risk = 'caution'
        risk_color = 'yellow'
    elif angle < thresholds['high_risk'][1]:
        risk = 'high'
        risk_color = 'orange'
    else:
        risk = 'critical'
        risk_color = 'red'
    
    # Special handling for overhead smash
    if stroke_type == 'overhead':
        overhead_range = thresholds['overhead_exception']
        if overhead_range[0] <= angle <= overhead_range[1]:
            risk = 'safe'  # Override - acceptable for overhead
            risk_color = 'green'
            within_safe_range = True
        else:
            within_safe_range = False
    else:
        within_safe_range = angle < thresholds['safe'][1]
    
    return {
        'angle': round(angle, 1),
        'risk_level': risk,
        'risk_color': risk_color,
        'within_safe_range': within_safe_range,
        'threshold_safe': thresholds['safe'],
        'threshold_caution': thresholds['caution']
    }


def calculate_elbow_flexion_validated(shoulder, elbow, wrist, stroke_type: str = 'groundstroke') -> Dict[str, Any]:
    """
    Calculate elbow flexion with stroke-specific optimal ranges.
    
    Args:
        shoulder, elbow, wrist: MediaPipe landmarks
        stroke_type: Type of stroke being performed
    
    Returns:
        Dictionary with angle and validation info
    """
    angle = calculate_angle(shoulder, elbow, wrist)
    
    # Get optimal range for this stroke
    optimal_range = ANGLE_THRESHOLDS['elbow_flexion'].get(stroke_type, (90, 160))
    
    within_optimal = optimal_range[0] <= angle <= optimal_range[1]
    
    # Provide feedback
    if angle < optimal_range[0]:
        feedback = f"Elbow too bent (< {optimal_range[0]}°)"
    elif angle > optimal_range[1]:
        feedback = f"Elbow over-extended (> {optimal_range[1]}°)"
    else:
        feedback = "Optimal elbow angle"
    
    return {
        'angle': round(angle, 1),
        'optimal_range': optimal_range,
        'within_optimal': within_optimal,
        'feedback': feedback
    }


def calculate_knee_flexion_validated(hip, knee, ankle) -> Dict[str, Any]:
    """
    Calculate knee flexion with stress indicators.
    
    Args:
        hip, knee, ankle: MediaPipe landmarks
    
    Returns:
        Dictionary with angle and stress indicators
    """
    angle = calculate_angle(hip, knee, ankle)
    
    safe_max = ANGLE_THRESHOLDS['knee_flexion']['safe_max']
    
    # Determine stress level
    if angle > safe_max:
        stress_level = 'high'
        feedback = f"Deep knee flexion (>{safe_max}°) - patellar stress risk"
    elif angle < 20:
        stress_level = 'low'
        feedback = "Legs too straight - poor athletic stance"
    else:
        stress_level = 'normal'
        feedback = "Good athletic stance"
    
    # Check if in optimal ready position
    ready_range = ANGLE_THRESHOLDS['knee_flexion']['ready']
    in_ready_position = ready_range[0] <= angle <= ready_range[1]
    
    return {
        'angle': round(angle, 1),
        'stress_level': stress_level,
        'in_ready_position': in_ready_position,
        'feedback': feedback
    }


def calculate_hip_rotation(left_hip, right_hip) -> float:
    """
    Calculate hip rotation from shoulder alignment.
    
    Args:
        left_hip, right_hip: MediaPipe landmarks
    
    Returns:
        Rotation angle in degrees
    """
    if not all([left_hip, right_hip]):
        return 0.0
    
    # Calculate horizontal distance
    dx = abs(right_hip.x - left_hip.x)
    dy = abs(right_hip.y - left_hip.y)
    
    # Calculate rotation angle from horizontal plane
    # More rotation = larger y difference relative to x
    if dx > 0:
        rotation = np.degrees(np.arctan(dy / dx))
    else:
        rotation = 0.0
    
    return float(rotation)


def calculate_hip_rotation_validated(left_hip, right_hip, stroke_type: str = 'groundstroke') -> Dict[str, Any]:
    """
    Calculate hip rotation with power generation assessment.
    
    Args:
        left_hip, right_hip: MediaPipe landmarks
        stroke_type: Type of stroke being performed
    
    Returns:
        Dictionary with rotation and power generation info
    """
    rotation = calculate_hip_rotation(left_hip, right_hip)
    
    # Get optimal range for this stroke
    optimal_range = ANGLE_THRESHOLDS['hip_rotation'].get(stroke_type, (0, 45))
    power_threshold = ANGLE_THRESHOLDS['hip_rotation']['power_threshold']
    
    # Assess power generation
    if rotation >= optimal_range[0]:
        power_generation = 'good'
        feedback = "Good hip rotation for power"
    elif rotation >= power_threshold:
        power_generation = 'moderate'
        feedback = "Moderate hip rotation"
    else:
        power_generation = 'poor'
        feedback = "Insufficient hip rotation - using arm only"
    
    within_optimal = optimal_range[0] <= rotation <= optimal_range[1]
    
    return {
        'angle': round(rotation, 1),
        'optimal_range': optimal_range,
        'within_optimal': within_optimal,
        'power_generation': power_generation,
        'feedback': feedback
    }


def calculate_wrist_position(mid_hip_x: float, mid_hip_y: float, wrist) -> Tuple[float, float, float]:
    """
    Calculate wrist position relative to body center.
    
    Args:
        mid_hip_x, mid_hip_y: Body center coordinates
        wrist: MediaPipe wrist landmark
    
    Returns:
        (dx, dy, distance_normalized)
    """
    if not wrist:
        return (0.0, 0.0, 0.0)
    
    dx = wrist.x - mid_hip_x
    dy = wrist.y - mid_hip_y
    
    # Normalized distance
    dist = np.sqrt(dx**2 + dy**2)
    
    return (float(dx), float(dy), float(dist))


def calculate_biomechanics_for_stroke(keypoints: Dict, stroke_type: str) -> Dict[str, Any]:
    """
    Calculate comprehensive biomechanics for a specific stroke type.
    
    Args:
        keypoints: Dictionary of MediaPipe landmarks
        stroke_type: Type of stroke being performed
    
    Returns:
        Dictionary of validated biomechanics metrics
    """
    results = {}
    
    # Shoulder abduction
    if all(k in keypoints for k in ['right_shoulder', 'right_elbow', 'right_hip']):
        results['shoulder_abduction'] = calculate_shoulder_abduction_validated(
            keypoints['right_shoulder'],
            keypoints['right_elbow'],
            keypoints['right_hip'],
            stroke_type
        )
    
    # Elbow flexion
    if all(k in keypoints for k in ['right_shoulder', 'right_elbow', 'right_wrist']):
        results['elbow_flexion'] = calculate_elbow_flexion_validated(
            keypoints['right_shoulder'],
            keypoints['right_elbow'],
            keypoints['right_wrist'],
            stroke_type
        )
    
    # Knee flexion
    if all(k in keypoints for k in ['right_hip', 'right_knee', 'right_ankle']):
        results['knee_flexion'] = calculate_knee_flexion_validated(
            keypoints['right_hip'],
            keypoints['right_knee'],
            keypoints['right_ankle']
        )
    
    # Hip rotation
    if all(k in keypoints for k in ['left_hip', 'right_hip']):
        results['hip_rotation'] = calculate_hip_rotation_validated(
            keypoints['left_hip'],
            keypoints['right_hip'],
            stroke_type
        )
    
    # Wrist position
    if all(k in keypoints for k in ['left_hip', 'right_hip', 'right_wrist']):
        mid_hip_x = (keypoints['left_hip'].x + keypoints['right_hip'].x) / 2.0
        mid_hip_y = (keypoints['left_hip'].y + keypoints['right_hip'].y) / 2.0
        
        dx, dy, dist = calculate_wrist_position(mid_hip_x, mid_hip_y, keypoints['right_wrist'])
        
        results['wrist_position'] = {
            'distance_from_center': round(dist, 3),
            'vector_x': round(dx, 3),
            'vector_y': round(dy, 3)
        }
    
    return results
