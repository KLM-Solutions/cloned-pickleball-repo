/**
 * Generate coaching feedback based on stroke type and metrics
 */

import { FrameMetrics, StrokeType } from './types';

/**
 * Generate feedback for a single frame based on stroke type
 */
export function generateFeedback(
  metrics: FrameMetrics,
  strokeType: StrokeType
): string[] {
  const feedback: string[] = [];
  
  switch (strokeType) {
    case 'serve':
      feedback.push(...generateServeFeedback(metrics));
      break;
    case 'dink':
      feedback.push(...generateDinkFeedback(metrics));
      break;
    case 'groundstroke':
      feedback.push(...generateGroundstrokeFeedback(metrics));
      break;
    case 'overhead':
      feedback.push(...generateOverheadFeedback(metrics));
      break;
    case 'volley':
      feedback.push(...generateVolleyFeedback(metrics));
      break;
  }
  
  return feedback;
}

/**
 * Serve-specific feedback
 * Key checks:
 * - Contact point must be below waist (wrist_above_waist should be false)
 * - Hip rotation for power
 */
function generateServeFeedback(metrics: FrameMetrics): string[] {
  const feedback: string[] = [];
  
  // Check contact point (serve must be below waist)
  if (metrics.wrist_above_waist) {
    feedback.push('FAULT: Contact point too high (Above waist)');
  }
  
  // Check hip rotation for power
  const hipRotation = metrics.hip_rotation_deg || 0;
  if (hipRotation < 10) {
    feedback.push('Power: Rotate hips more before contact');
  }
  
  // Check elbow angle
  const elbowAngle = metrics.right_elbow_flexion || 0;
  if (elbowAngle < 90 || elbowAngle > 130) {
    feedback.push(`Elbow angle (${elbowAngle}°) - optimal is 90-130° for serve`);
  }
  
  return feedback;
}

/**
 * Dink-specific feedback
 * Key checks:
 * - Low body position (knee flexion)
 * - Minimal shoulder movement
 */
function generateDinkFeedback(metrics: FrameMetrics): string[] {
  const feedback: string[] = [];
  
  // Check knee flexion (should be bent for low position)
  const kneeFlexion = metrics.right_knee_flexion || 180;
  if (kneeFlexion > 150) {
    feedback.push('Form: Get Lower! Bend your knees, not just your back.');
  }
  
  // Check shoulder - should be low for dink
  const shoulderAngle = metrics.right_shoulder_abduction || 0;
  if (shoulderAngle > 40) {
    feedback.push('Form: Keep paddle and arm lower for better control');
  }
  
  return feedback;
}

/**
 * Groundstroke-specific feedback
 * Key checks:
 * - Hip rotation for power
 * - Shoulder in moderate range
 */
function generateGroundstrokeFeedback(metrics: FrameMetrics): string[] {
  const feedback: string[] = [];
  
  // Check hip rotation
  const hipRotation = metrics.hip_rotation_deg || 0;
  if (hipRotation < 30) {
    feedback.push('Power: Engage hips more - rotate through the shot');
  }
  
  // Check shoulder abduction
  const shoulderAngle = metrics.right_shoulder_abduction || 0;
  if (shoulderAngle < 60) {
    feedback.push('Form: Raise elbow for better power generation');
  } else if (shoulderAngle > 140) {
    feedback.push('Caution: Shoulder too high - risk of strain');
  }
  
  return feedback;
}

/**
 * Overhead-specific feedback
 * Key checks:
 * - Arm extension at contact
 * - Wrist above head
 */
function generateOverheadFeedback(metrics: FrameMetrics): string[] {
  const feedback: string[] = [];
  
  // Check elbow extension
  const elbowAngle = metrics.right_elbow_flexion || 0;
  if (elbowAngle < 150) {
    feedback.push('Power: Extend arm fully at contact');
  }
  
  // Check if wrist is above head (good for overhead)
  if (!metrics.wrist_above_head) {
    feedback.push('Timing: Contact point should be above head');
  }
  
  // Check knee flexion for loading
  const kneeFlexion = metrics.right_knee_flexion || 180;
  if (kneeFlexion > 160) {
    feedback.push('Power: Load legs more before swing');
  }
  
  return feedback;
}

/**
 * Volley-specific feedback
 * Key checks:
 * - Compact motion (moderate shoulder)
 * - Ready position (knee bend)
 */
function generateVolleyFeedback(metrics: FrameMetrics): string[] {
  const feedback: string[] = [];
  
  // Check shoulder range (should be moderate for volley)
  const shoulderAngle = metrics.right_shoulder_abduction || 0;
  if (shoulderAngle > 90) {
    feedback.push('Form: Keep motion compact - less backswing');
  } else if (shoulderAngle < 40) {
    feedback.push('Form: Paddle too low - raise to ready position');
  }
  
  // Check ready position
  const kneeFlexion = metrics.right_knee_flexion || 180;
  if (kneeFlexion > 160) {
    feedback.push('Stance: Bend knees for better ready position');
  }
  
  return feedback;
}

/**
 * Generate session-level recommendations based on risk analysis
 */
export function generateSessionRecommendations(
  riskPercentages: { shoulder_overuse: number; poor_kinetic_chain: number; knee_stress: number },
  strokeType: StrokeType
): string[] {
  const recommendations: string[] = [];
  
  if (riskPercentages.shoulder_overuse > 10) {
    recommendations.push('⚠️ High shoulder strain detected. Focus on reducing arm-only swings.');
  }
  
  if (riskPercentages.poor_kinetic_chain > 20) {
    recommendations.push('💡 Improve power by engaging hips and core more in your swings.');
  }
  
  if (riskPercentages.knee_stress > 15) {
    recommendations.push('🦵 Consider knee strengthening exercises. Avoid excessive squatting.');
  }
  
  if (recommendations.length === 0) {
    recommendations.push('✅ Good form detected! Keep up the consistent technique.');
  }
  
  return recommendations;
}

