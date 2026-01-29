"""
Injury risk detection based on biomechanics research.
Analyzes frame-by-frame and provides session summaries.
"""
from typing import Dict, List, Any


class InjuryRiskDetector:
    """
    Detects injury risks based on biomechanical patterns.
    
    Based on research findings:
    - Shoulder abduction >140° = high risk (except overhead)
    - Poor kinetic chain (no hip rotation) = elbow/back strain
    - Excessive knee flexion >90° = patellar stress
    - Repetitive high-risk movements = overuse injury
    """
    
    def __init__(self):
        self.frame_history = []
        self.risk_counters = {
            'shoulder_overuse': 0,
            'elbow_strain': 0,
            'knee_stress': 0,
            'poor_kinetic_chain': 0
        }
        
        # Thresholds from research
        self.thresholds = {
            'shoulder_overuse_percentage': 10,  # % of frames
            'poor_kinetic_chain_percentage': 20,
            'knee_stress_percentage': 15,
            'shoulder_critical_angle': 160,
            'shoulder_high_angle': 140,
            'knee_stress_angle': 90,
            'hip_rotation_minimum': 30  # For power strokes
        }
    
    def analyze_frame(self, biomechanics: Dict[str, Any], stroke_type: str) -> List[Dict[str, Any]]:
        """
        Analyze single frame for injury risks.
        
        Args:
            biomechanics: Validated biomechanics from angles.py
            stroke_type: Type of stroke being performed
        
        Returns:
            List of detected risks for this frame
        """
        risks = []
        
        # 1. SHOULDER OVERUSE
        shoulder = biomechanics.get('shoulder_abduction', {})
        shoulder_angle = shoulder.get('angle', 0)
        risk_level = shoulder.get('risk_level', 'safe')
        
        if risk_level in ['high', 'critical'] and stroke_type != 'overhead':
            # High shoulder abduction outside of overhead smash
            severity = 'critical' if risk_level == 'critical' else 'high'
            
            risks.append({
                'type': 'shoulder_overuse',
                'severity': severity,
                'angle': shoulder_angle,
                'stroke': stroke_type,
                'message': f'Shoulder abduction {shoulder_angle}° exceeds safe range',
                'recommendation': 'Reduce shoulder abduction to <140°. Focus on hip rotation for power.'
            })
            
            self.risk_counters['shoulder_overuse'] += 1
        
        # 2. POOR KINETIC CHAIN (No hip rotation)
        hip = biomechanics.get('hip_rotation', {})
        hip_rotation = hip.get('angle', 0)
        power_gen = hip.get('power_generation', 'good')
        
        if power_gen == 'poor' and stroke_type in ['groundstroke', 'serve', 'overhead']:
            # Power strokes without hip rotation = arm-only swing
            risks.append({
                'type': 'poor_kinetic_chain',
                'severity': 'medium',
                'hip_rotation': hip_rotation,
                'stroke': stroke_type,
                'message': f'Insufficient hip rotation ({hip_rotation}°)',
                'recommendation': 'Engage hips and core for power. Reduce arm strain.'
            })
            
            self.risk_counters['poor_kinetic_chain'] += 1
            self.risk_counters['elbow_strain'] += 1  # Consequence
        
        # 3. KNEE STRESS (Excessive flexion)
        knee = biomechanics.get('knee_flexion', {})
        knee_angle = knee.get('angle', 180)
        stress_level = knee.get('stress_level', 'normal')
        
        if stress_level == 'high':
            risks.append({
                'type': 'knee_stress',
                'severity': 'medium',
                'angle': knee_angle,
                'message': f'Deep knee flexion ({knee_angle}°) detected',
                'recommendation': 'Avoid excessive squatting. Maintain athletic stance (20-45°).'
            })
            
            self.risk_counters['knee_stress'] += 1
        
        # 4. ELBOW STRAIN (Poor technique)
        elbow = biomechanics.get('elbow_flexion', {})
        if not elbow.get('within_optimal', True):
            # Elbow outside optimal range for stroke
            self.risk_counters['elbow_strain'] += 1
        
        # Store frame data
        self.frame_history.append({
            'biomechanics': biomechanics,
            'risks': risks,
            'stroke_type': stroke_type
        })
        
        return risks
    
    def get_session_summary(self) -> Dict[str, Any]:
        """
        Get overall injury risk summary for the session.
        
        Returns:
            Dictionary with risk analysis and recommendations
        """
        total_frames = len(self.frame_history)
        if total_frames == 0:
            return {
                'total_frames': 0,
                'overall_risk': 'unknown',
                'alerts': [],
                'recommendations': []
            }
        
        # Calculate percentages
        shoulder_pct = (self.risk_counters['shoulder_overuse'] / total_frames) * 100
        kinetic_chain_pct = (self.risk_counters['poor_kinetic_chain'] / total_frames) * 100
        knee_pct = (self.risk_counters['knee_stress'] / total_frames) * 100
        elbow_pct = (self.risk_counters['elbow_strain'] / total_frames) * 100
        
        alerts = []
        recommendations = []
        
        # SHOULDER OVERUSE ALERT
        if shoulder_pct > self.thresholds['shoulder_overuse_percentage']:
            alerts.append({
                'type': 'shoulder_overuse',
                'severity': 'high',
                'percentage': round(shoulder_pct, 1),
                'message': f'{shoulder_pct:.1f}% of frames show high shoulder risk',
                'icon': '⚠️'
            })
            
            recommendations.append({
                'priority': 'high',
                'category': 'Injury Prevention',
                'title': 'Reduce Shoulder Strain',
                'description': 'Excessive shoulder abduction detected. Focus on proper technique.',
                'actions': [
                    'Practice shoulder rotation drills',
                    'Reduce overhead smash frequency',
                    'Strengthen rotator cuff muscles',
                    'Consider professional coaching for form correction'
                ]
            })
        
        # POOR KINETIC CHAIN ALERT
        if kinetic_chain_pct > self.thresholds['poor_kinetic_chain_percentage']:
            alerts.append({
                'type': 'technique',
                'severity': 'medium',
                'percentage': round(kinetic_chain_pct, 1),
                'message': f'Insufficient hip rotation in {kinetic_chain_pct:.1f}% of strokes',
                'icon': '⚠️'
            })
            
            recommendations.append({
                'priority': 'medium',
                'category': 'Technique',
                'title': 'Improve Power Generation',
                'description': 'Using arm-only swings increases injury risk and reduces power.',
                'actions': [
                    'Practice weight transfer drills',
                    'Focus on hip rotation before arm swing',
                    'Strengthen core muscles',
                    'Watch tutorial videos on kinetic chain'
                ]
            })
        
        # KNEE STRESS ALERT
        if knee_pct > self.thresholds['knee_stress_percentage']:
            alerts.append({
                'type': 'knee_stress',
                'severity': 'medium',
                'percentage': round(knee_pct, 1),
                'message': f'Excessive knee flexion in {knee_pct:.1f}% of frames',
                'icon': '⚠️'
            })
            
            recommendations.append({
                'priority': 'medium',
                'category': 'Form',
                'title': 'Protect Your Knees',
                'description': 'Deep squatting increases patellar tendinitis risk.',
                'actions': [
                    'Maintain athletic stance (knees bent 20-45°)',
                    'Avoid excessive squatting during dinks',
                    'Strengthen quadriceps and hamstrings',
                    'Consider knee support if pain persists'
                ]
            })
        
        # ELBOW STRAIN (from poor technique)
        if elbow_pct > 15:  # 15% threshold
            recommendations.append({
                'priority': 'low',
                'category': 'Technique',
                'title': 'Elbow Positioning',
                'description': 'Elbow angles outside optimal range detected.',
                'actions': [
                    'Focus on proper elbow extension for each stroke',
                    'Avoid tight grip (causes elbow strain)',
                    'Practice with lighter paddle if experiencing pain'
                ]
            })
        
        # Determine overall risk level
        if len(alerts) >= 2:
            overall_risk = 'high'
        elif len(alerts) == 1:
            overall_risk = 'medium'
        else:
            overall_risk = 'low'
        
        # Add positive feedback if low risk
        if overall_risk == 'low':
            recommendations.append({
                'priority': 'info',
                'category': 'Great Job!',
                'title': 'Excellent Biomechanics',
                'description': 'Your form shows low injury risk. Keep it up!',
                'actions': [
                    'Continue current training routine',
                    'Maintain proper warm-up and cool-down',
                    'Stay consistent with technique'
                ]
            })
        
        return {
            'total_frames': total_frames,
            'risk_counters': self.risk_counters,
            'percentages': {
                'shoulder_overuse': round(shoulder_pct, 1),
                'poor_kinetic_chain': round(kinetic_chain_pct, 1),
                'knee_stress': round(knee_pct, 1),
                'elbow_strain': round(elbow_pct, 1)
            },
            'alerts': alerts,
            'recommendations': recommendations,
            'overall_risk': overall_risk
        }
    
    def reset(self):
        """Reset detector for new session."""
        self.frame_history = []
        self.risk_counters = {
            'shoulder_overuse': 0,
            'elbow_strain': 0,
            'knee_stress': 0,
            'poor_kinetic_chain': 0
        }
